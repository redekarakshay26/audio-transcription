import subprocess
import os

input_file  = "rajspeech.mp3"
output_file = os.path.splitext(input_file)[0] + ".webm"

cmd = [
    "ffmpeg",
    "-hwaccel", "cuda",          # GPU accelerated decoding
    "-i", input_file,
    "-c:a", "libopus",           # Opus codec for .webm
    "-b:a", "128k",
    "-y",                        # overwrite output if exists
    output_file
]

subprocess.run(cmd, check=True)
print(f"Done: {output_file}")