import os
import time
import warnings
import concurrent.futures
from typing import List, Dict, Tuple

import numpy as np
import torch
import whisperx
from dotenv import load_dotenv
from whisperx.diarize import DiarizationPipeline

# ────────────────────────────────────────────────────────────────
# Patch torch.load early to allow loading pyannote models
# (required for PyTorch ≥ 2.6 due to weights_only=True default)
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load
# ────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchaudio.*backend.*", category=UserWarning)

load_dotenv()

MODEL_SIZE   = "base"
DEVICE       =  "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float32"
LANGUAGE     = "en"
SAMPLE_RATE  = 16000
BATCH_SIZE   = 4
CHUNK_SEC    = 120
OVERLAP_SEC  = 15
MAX_WORKERS  = 4          # Adjust based on your CPU cores / RAM
HF_TOKEN     = os.getenv("HF_TOKEN_WHISPERX", "").strip()


class AudioTranscriber:
    def __init__(self):
        self.asr_model      = None
        self.align_model    = None
        self.align_metadata = None
        self.diarize_pipe   = None

    def load_models(self):
        if self.asr_model is None:
            print("--- Loading WhisperX transcription model")
            self.asr_model = whisperx.load_model(
                MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE
            )
        if self.align_model is None:
            print("--- Loading alignment model")
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=LANGUAGE, device=DEVICE
            )
        if self.diarize_pipe is None:
            print("--- Loading pyannote speaker diarization 3.0")
            self.diarize_pipe = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.0",
                use_auth_token=HF_TOKEN if HF_TOKEN else True,
                device=DEVICE,
            )

    def split_audio(self, audio: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        chunk_samples = CHUNK_SEC * SAMPLE_RATE
        overlap_samples = OVERLAP_SEC * SAMPLE_RATE
        if overlap_samples >= chunk_samples:
            raise ValueError("OVERLAP_SEC must be smaller than CHUNK_SEC")
        step_samples = chunk_samples - overlap_samples

        chunks = []
        for start in range(0, len(audio), step_samples):
            end = min(start + chunk_samples, len(audio))
            chunks.append((audio[start:end], start / SAMPLE_RATE))
            if end >= len(audio):
                break
        return chunks

    def transcribe_chunk(self, chunk: np.ndarray, chunk_idx: int, offset_sec: float) -> Tuple[List[Dict], float]:
        print(f"--- [Chunk {chunk_idx}] Transcribing | offset={offset_sec:.1f}s | duration={len(chunk)/SAMPLE_RATE:.1f}s")

        try:
            result = self.asr_model.transcribe(chunk, batch_size=BATCH_SIZE)
            result = whisperx.align(
                result["segments"], self.align_model, self.align_metadata, chunk, DEVICE
            )

            segments = result["segments"]
            # Crucial fix: discard segments born in overlap zone (except chunk 0)
            if chunk_idx > 0:
                segments = [seg for seg in segments if seg['start'] >= OVERLAP_SEC]

            print(f"--- [Chunk {chunk_idx}] Done | {len(segments)} segments after overlap trim")
            return segments, offset_sec

        except Exception as e:
            print(f"--- [Chunk {chunk_idx}] Failed: {str(e)}")
            return [], offset_sec

    def process_audio(self, audio_file_path: str) -> List[Dict]:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        overall_start = time.time()
        print(f"--- Loading audio: {os.path.basename(audio_file_path)}")

        try:
            audio = whisperx.load_audio(audio_file_path, sr=SAMPLE_RATE)
        except Exception as e:
            raise ValueError(f"Failed to load audio: {str(e)}")

        if len(audio) == 0:
            print("--- Empty audio file detected")
            return []

        print(f"--- Audio duration: {len(audio)/SAMPLE_RATE/60:.1f} minutes")

        self.load_models()

        chunks = self.split_audio(audio)
        print(f"--- Audio split into {len(chunks)} chunk(s)")

        # Start transcription in background threads
        print("--- Submitting transcription chunks in background")
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunk_futures = {
                executor.submit(self.transcribe_chunk, chunk_audio, idx, offset): idx
                for idx, (chunk_audio, offset) in enumerate(chunks)
            }

            # Diarization in main thread (avoids known pyannote threading issues)
            print("--- [Diarization] Started on full audio")
            try:
                diarize_segments = self.diarize_pipe(audio)
                print(f"--- [Diarization] Done | {len(diarize_segments)} speaker turns")
            except Exception as e:
                print(f"--- [Diarization] Failed: {str(e)}. Continuing without speaker labels.")
                diarize_segments = None

            # Collect transcription results
            chunk_results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(chunk_futures):
                idx = chunk_futures[future]
                try:
                    chunk_results[idx] = future.result()
                except Exception as e:
                    print(f"--- [Chunk {idx}] Collection failed: {e}")
                    chunk_results[idx] = ([], 0)

        # Merge transcribed segments with global timestamps
        all_transcribed = []
        for local_segments, offset in chunk_results:
            for seg in local_segments:
                seg_copy = {**seg}
                seg_copy["start"] += offset
                seg_copy["end"] += offset
                if 'words' in seg_copy:
                    for word in seg_copy['words']:
                        if 'start' in word:
                            word['start'] += offset
                        if 'end' in word:
                            word['end'] += offset
                all_transcribed.append(seg_copy)

        all_transcribed.sort(key=lambda x: x['start'])
        print(f"--- Transcription merged | {len(all_transcribed)} total segments")

        # Assign speaker labels if diarization succeeded
        print("--- Assigning speaker labels")
        if diarize_segments is not None:
            diarized_result = whisperx.assign_word_speakers(
                diarize_segments, {"segments": all_transcribed}
            )
            segments_to_use = diarized_result["segments"]
        else:
            segments_to_use = [
                {"start": s["start"], "end": s["end"], "text": s.get("text", "").strip(), "speaker": "UNKNOWN"}
                for s in all_transcribed
            ]

        # Final lightweight cleanup (should rarely trigger now)
        final_segments = []
        last_end = -1.0
        for seg in segments_to_use:
            start = round(seg["start"], 2)
            end = round(seg["end"], 2)
            text = seg.get("text", "").strip()
            if not text:
                continue
            speaker = seg.get("speaker", "UNKNOWN")

            if start >= last_end:
                final_segments.append({
                    "speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": text,
                })
                last_end = end
            elif end > last_end:
                # Rare partial overlap — trim start
                final_segments.append({
                    "speaker": speaker,
                    "start": last_end,
                    "end": end,
                    "text": text,
                })
                last_end = end

        print(f"--- Done | {len(final_segments)} final segments | total time: {time.time() - overall_start:.1f}s")
        return final_segments


if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "misunderstood.webm")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    transcriber = AudioTranscriber()
    segments = transcriber.process_audio(input_file)

    print("\nFirst 10 segments:")
    for seg in segments[:10]:
        print(f'{seg["speaker"]:>12} [{seg["start"]:6.2f} → {seg["end"]:6.2f}] {seg["text"]}')