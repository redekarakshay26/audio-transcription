
import os
import time
import warnings
from typing import List, Dict
from datetime import datetime

import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

_orig_torch_load = torch.load
torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchaudio.*backend.*", category=UserWarning)

load_dotenv()

MODEL_SIZE   = "base"
# DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE       = "cuda" 
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"
LANGUAGE     = "en"
SAMPLE_RATE  = 16000
# BATCH_SIZE   = 16 if DEVICE == "cuda" else 4
BATCH_SIZE   = 32
# HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
HF_TOKEN     = os.getenv("HF_TOKEN_WHISPERX", "").strip()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True


def get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"available": False}
    idx   = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return {
        "available"    : True,
        "name"         : props.name,
        "vram_total"   : f"{props.total_memory / 1024**3:.2f} GB",
        "cuda_version" : torch.version.cuda,
        "cudnn_version": str(torch.backends.cudnn.version()),
        "device_index" : idx,
    }

GPU_INFO = get_gpu_info()
if GPU_INFO["available"]:
    print(f"--- [GPU] {GPU_INFO['name']} | VRAM={GPU_INFO['vram_total']} | CUDA={GPU_INFO['cuda_version']} | cuDNN={GPU_INFO['cudnn_version']}")
else:
    print("--- [GPU] CUDA not available → falling back to CPU")


class AudioTranscriber:
    def __init__(self):
        self.asr_model      = None
        self.align_model    = None
        self.align_metadata = None
        self.diarize_pipe   = None

    def load_models(self):
        if self.asr_model is None:
            print("--- [Model Load] Loading WhisperX ASR model...")
            t0 = time.time()
            self.asr_model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)
            print(f"--- [Model Load] ASR model ready | {time.time()-t0:.2f}s")

        if self.align_model is None:
            print("--- [Model Load] Loading alignment model...")
            t0 = time.time()
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)
            print(f"--- [Model Load] Alignment model ready | {time.time()-t0:.2f}s")

        if self.diarize_pipe is None:
            print("--- [Model Load] Loading pyannote diarization 3.0...")
            t0 = time.time()
            self.diarize_pipe = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=HF_TOKEN if HF_TOKEN else True,
                device=DEVICE,
            )
            print(f"--- [Model Load] Diarization model ready | {time.time()-t0:.2f}s")

    def process_audio(self, audio_file_path: str) -> List[Dict]:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"File not found: {audio_file_path}")

        timings     = {}
        total_start = time.time()

        # ── Load audio ───────────────────────────────────────────────────────
        print(f"--- [Audio] Loading file: {audio_file_path}")
        t0    = time.time()
        audio = whisperx.load_audio(audio_file_path, sr=SAMPLE_RATE)
        timings["audio_load"] = round(time.time() - t0, 2)

        if len(audio) == 0:
            print("--- [Audio] Empty audio, aborting.")
            return []

        duration_sec = len(audio) / SAMPLE_RATE
        duration_min = duration_sec / 60
        print(f"--- [Audio] Loaded | duration={duration_sec:.1f}s ({duration_min:.2f}m) | load_time={timings['audio_load']}s")

        self.load_models()

        # ── Transcription ────────────────────────────────────────────────────
        print(f"--- [Transcription] Started | batch_size={BATCH_SIZE} | device={DEVICE} | compute_type={COMPUTE_TYPE}")
        t0 = time.time()
        try:
            result = self.asr_model.transcribe(audio, batch_size=BATCH_SIZE)
            timings["transcription"] = round(time.time() - t0, 2)
            print(f"--- [Transcription] Done | {len(result['segments'])} segments | {timings['transcription']}s ({timings['transcription']/60:.2f}m)")
        except Exception as e:
            print(f"--- [Transcription] Failed: {e}")
            return []

        # ── Alignment ────────────────────────────────────────────────────────
        print("--- [Alignment] Started | aligning words to timestamps")
        t0 = time.time()
        try:
            aligned = whisperx.align(result["segments"], self.align_model, self.align_metadata, audio, DEVICE)
            timings["alignment"] = round(time.time() - t0, 2)
            print(f"--- [Alignment] Done | {timings['alignment']}s ({timings['alignment']/60:.2f}m)")
            transcribed_segments = aligned["segments"]
        except Exception as e:
            print(f"--- [Alignment] Failed: {e}")
            return []

        # ── Diarization ──────────────────────────────────────────────────────
        print("--- [Diarization] Started on full audio")
        t0 = time.time()
        try:
            diarize_segments = self.diarize_pipe(audio)
            timings["diarization"] = round(time.time() - t0, 2)
            print(f"--- [Diarization] Done | {len(diarize_segments)} speaker turns found | {timings['diarization']}s ({timings['diarization']/60:.2f}m)")
        except Exception as e:
            print(f"--- [Diarization] Failed: {e}. Using UNKNOWN speakers.")
            diarize_segments = None
            timings["diarization"] = round(time.time() - t0, 2)

        # ── Speaker Assignment ───────────────────────────────────────────────
        print("--- [Speaker Assignment] Assigning speakers to segments")
        t0 = time.time()
        if diarize_segments is not None:
            diarized        = whisperx.assign_word_speakers(diarize_segments, {"segments": transcribed_segments})
            segments_to_use = diarized["segments"]
        else:
            segments_to_use = transcribed_segments
        timings["speaker_assignment"] = round(time.time() - t0, 2)
        print(f"--- [Speaker Assignment] Done | {timings['speaker_assignment']}s")

        # ── Final cleanup ────────────────────────────────────────────────────
        final_segments: List[Dict] = []
        last_end = -1.0
        for seg in segments_to_use:
            start = round(seg.get("start", 0), 2)
            end   = round(seg.get("end", 0), 2)
            text  = seg.get("text", "").strip()
            if not text or end <= start:
                continue
            speaker = seg.get("speaker", "UNKNOWN")
            if start >= last_end:
                final_segments.append({"speaker": speaker, "start": start, "end": end, "text": text})
                last_end = end
            elif end > last_end:
                final_segments.append({"speaker": speaker, "start": last_end, "end": end, "text": text})
                last_end = end

        timings["total"] = round(time.time() - total_start, 2)
        print(f"--- [Done] Pipeline complete | {len(final_segments)} segments | total={timings['total']}s ({timings['total']/60:.2f}m)")

        self._save_log(audio_file_path, duration_sec, duration_min, timings, final_segments)
        return final_segments

    def _save_log(self, audio_file_path: str, duration_sec: float, duration_min: float, timings: dict, segments: List[Dict]):
        out_dir  = os.path.dirname(os.path.abspath(audio_file_path))
        base     = os.path.splitext(os.path.basename(audio_file_path))[0]
        diarize_tag = "pyannote" + "pyannote/speaker-diarization-3.1".split("diarization-")[-1]
        log_path = os.path.join(out_dir, f"{base}_transcript_{DEVICE.lower()}_{BATCH_SIZE}_{diarize_tag}_.txt")
        now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 65 + "\n")
            f.write(f"  Run Timestamp    : {now}\n")
            f.write("=" * 65 + "\n")

            f.write("\n[Config]\n")
            f.write(f"  model_size       : {MODEL_SIZE}\n")
            f.write(f"  device           : {DEVICE}\n")
            f.write(f"  compute_type     : {COMPUTE_TYPE}\n")
            f.write(f"  language         : {LANGUAGE}\n")
            f.write(f"  batch_size       : {BATCH_SIZE}\n")
            f.write(f"  diarize_model    : pyannote/speaker-diarization-3.1\n")

            f.write("\n[GPU]\n")
            if GPU_INFO.get("available"):
                f.write(f"  name             : {GPU_INFO['name']}\n")
                f.write(f"  vram_total       : {GPU_INFO['vram_total']}\n")
                f.write(f"  cuda_version     : {GPU_INFO['cuda_version']}\n")
                f.write(f"  cudnn_version    : {GPU_INFO['cudnn_version']}\n")
                f.write(f"  device_index     : {GPU_INFO['device_index']}\n")
            else:
                f.write("  GPU not available (running on CPU)\n")

            f.write("\n[Audio]\n")
            f.write(f"  file             : {os.path.basename(audio_file_path)}\n")
            f.write(f"  duration         : {duration_sec:.1f}s  ({duration_min:.2f}m)\n")
            f.write(f"  total segments   : {len(segments)}\n")

            f.write("\n[Timings]\n")
            f.write(f"  audio_load       : {timings['audio_load']}s  ({timings['audio_load']/60:.2f}m)\n")
            f.write(f"  transcription    : {timings['transcription']}s  ({timings['transcription']/60:.2f}m)\n")
            f.write(f"  alignment        : {timings['alignment']}s  ({timings['alignment']/60:.2f}m)\n")
            f.write(f"  diarization      : {timings['diarization']}s  ({timings['diarization']/60:.2f}m)\n")
            f.write(f"  speaker_assign   : {timings['speaker_assignment']}s  ({timings['speaker_assignment']/60:.2f}m)\n")
            f.write(f"  total            : {timings['total']}s  ({timings['total']/60:.2f}m)\n")

            f.write("\n[Transcript]\n")
            f.write("-" * 65 + "\n")
            for seg in segments:
                f.write(f"  {seg['speaker']:>12} [{seg['start']:6.2f} → {seg['end']:6.2f}] {seg['text']}\n")
            f.write("=" * 65 + "\n")

        print(f"--- [Log] Saved to: {log_path}")


if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rajspeech.webm")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    transcriber = AudioTranscriber()
    segments    = transcriber.process_audio(input_file)

    print("\nFirst 10 segments:")
    for seg in segments[:10]:
        print(f'{seg["speaker"]:>12} [{seg["start"]:6.2f} → {seg["end"]:6.2f}] {seg["text"]}')