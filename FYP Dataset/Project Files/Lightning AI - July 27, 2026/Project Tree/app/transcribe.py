import os
import sys
import gc
import json
import torch
import whisperx

from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------
# ENV & ARGS
# -----------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("HF_TOKEN not found!")
    sys.exit(1)

if len(sys.argv) < 2:
    print("Usage: python app/transcribe.py <run_dir>")
    sys.exit(1)

RUN_DIR      = sys.argv[1]
audio_path   = os.path.join(RUN_DIR, "audio", "audio.wav")
outputs_dir  = os.path.join(RUN_DIR, "outputs")
output_file  = os.path.join(outputs_dir, "transcript.json")

os.makedirs(outputs_dir, exist_ok=True)

if not os.path.exists(audio_path):
    print("Audio file not found!")
    sys.exit(1)

DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# -----------------------------------------
# UTILITY
# -----------------------------------------

def free_vram(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================================
# 1. TRANSCRIPTION
# =========================================

print("Loading WhisperX (small)...")

asr_model = whisperx.load_model(
    "small",
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

print("Transcribing...")

audio      = whisperx.load_audio(audio_path)
raw_result = asr_model.transcribe(audio, batch_size=16)

# =========================================
# 2. ALIGNMENT
# =========================================

print("Aligning words to audio...")

model_a, metadata = whisperx.load_align_model(
    language_code=raw_result["language"],
    device=DEVICE
)

aligned_result = whisperx.align(
    raw_result["segments"],
    model_a,
    metadata,
    audio,
    DEVICE
)

free_vram(asr_model, model_a)

# =========================================
# 3. DIARIZATION
# =========================================

print("Running diarization...")

diarize_model = whisperx.diarize.DiarizationPipeline(
    token=HF_TOKEN,
    device=DEVICE
)

diarize_segments = diarize_model(audio)

# =========================================
# 4. WORD-LEVEL ASSIGNMENT
# =========================================

print("Merging speakers with words...")

final_alignment = whisperx.assign_word_speakers(
    diarize_segments,
    aligned_result
)

free_vram(diarize_model)

# =========================================
# 5. BUILD SEGMENTS
# =========================================

unique_speakers = set()
segments        = []

for idx, seg in enumerate(
    final_alignment["segments"],
    start=1
):

    spk = seg.get("speaker", "UNKNOWN")

    unique_speakers.add(spk)

    segments.append({
        "id":         idx,
        "start":      round(seg["start"], 2),
        "end":        round(seg["end"],   2),
        "speaker_id": spk,
        "speaker":    spk,
        "text":       seg["text"].strip()
    })

# =========================================
# 6. SPEAKER MAP
# =========================================

speaker_names = {
    speaker: speaker
    for speaker in sorted(unique_speakers)
}



# =========================================
# 7. FINAL SPEAKER MAP
# =========================================

print(f"\nFinal speaker map:")

for spk_id, name in sorted(speaker_names.items()):
    print(f"  {spk_id} = {name}")


# =========================================
# 8. SAVE OUTPUT
# =========================================

with open(output_file, "w", encoding="utf-8") as f:

    json.dump(
        {
            "metadata": {
                "whisper_model": "whisperx_small",
                "device": DEVICE,
                "num_speakers": len(unique_speakers),
                "total_segments": len(segments),
                "processing_timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },

            "speakers": speaker_names,

            "segments": segments,
        },
        f,
        indent=4,
        ensure_ascii=False,
    )

print(f"\nTranscript saved to: {output_file}")