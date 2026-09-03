import os
import sys
import gc
import json
import torch

from PIL import Image
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationMixin,
    GenerationConfig
)

# -----------------------------------------
# INPUT
# -----------------------------------------

if len(sys.argv) < 2:
    print("Usage: python app/moondream_context.py <run_dir>")
    sys.exit(1)

RUN_DIR = sys.argv[1]

# -----------------------------------------
# PATHS
# -----------------------------------------

frames_dir = os.path.join(
    RUN_DIR,
    "frames"
)

outputs_dir = os.path.join(
    RUN_DIR,
    "outputs"
)

os.makedirs(
    outputs_dir,
    exist_ok=True
)

context_output = os.path.join(
    outputs_dir,
    "video_context.json"
)

# -----------------------------------------
# CHUNK DURATION (seconds)
# -----------------------------------------

CHUNK_DURATION = 30

# -----------------------------------------
# CHECK FRAMES DIRECTORY
# -----------------------------------------

if not os.path.exists(frames_dir):
    print("Frames directory not found!")
    sys.exit(1)

# =========================================
# 1. LOAD MOONDREAM — TRY MULTIPLE APPROACHES
# =========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "vikhyatk/moondream2"

model = None
tokenizer = None
use_builtin_tokenizer = False

# -----------------------------------------
# APPROACH 1: Latest revision (compatible with
# transformers >= 4.50 out of the box)
# -----------------------------------------

print("\nLoading Moondream2 (latest revision)...")

try:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision="2025-01-09",
        trust_remote_code=True,
        dtype=torch.float16
    ).to(DEVICE)

    # Newer revisions have tokenizer built in
    if hasattr(model, 'tokenizer'):
        tokenizer = model.tokenizer
        use_builtin_tokenizer = True
        print("Using built-in tokenizer.")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision="2025-01-09",
            trust_remote_code=True
        )

    # Quick test — try encode_image on a dummy
    print("Testing model API...")

    test_img = Image.new("RGB", (64, 64), color="black")
    test_enc = model.encode_image(test_img)
    _ = model.answer_question(
        test_enc, "test", tokenizer
    )

    print("Latest revision loaded and working!")
    REVISION = "2025-01-09"

except Exception as e:

    print(f"Latest revision failed: {e}")
    print("Falling back to 2024-08-26 with patch...")

    # -----------------------------------------
    # APPROACH 2: Old revision + full patch
    # -----------------------------------------

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    REVISION = "2024-08-26"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    # -----------------------------------------
    # FULL PATCH: GenerationMixin + GenerationConfig
    # -----------------------------------------

    def patch_model(obj):
        if obj is None:
            return
        if not isinstance(obj, GenerationMixin):
            obj.__class__ = type(
                obj.__class__.__name__,
                (obj.__class__, GenerationMixin),
                {}
            )
        if (
            not hasattr(obj, 'generation_config')
            or obj.generation_config is None
        ):
            obj.generation_config = GenerationConfig(
                max_new_tokens=256
            )

    patch_model(model)
    patch_model(
        getattr(model, 'text_model', None)
    )

    model = model.to(DEVICE)

    print("Patched model loaded.")

model.eval()

print("Moondream2 ready.\n")

# =========================================
# 2. COLLECT CHUNK DIRECTORIES
# =========================================

chunk_dirs = sorted([

    d for d in os.listdir(frames_dir)
    if os.path.isdir(
        os.path.join(frames_dir, d)
    )
])

if not chunk_dirs:
    print("No chunk directories found!")
    print("Check if preprocess.py extracted frames.")
    sys.exit(1)

print(f"Found {len(chunk_dirs)} chunks to process.\n")

# =========================================
# 3. VISION PROMPT
# =========================================

VISION_PROMPT = (
    "Describe the visual content of this frame concisely. "
    "Focus on: people visible, their actions, "
    "any text or slides on screen, "
    "the environment or setting. "
    "Keep it to 2-3 sentences."
)

# =========================================
# 4. PROCESS FRAMES BY CHUNK
# =========================================

chunks_data = []

total_frames = 0
failed_frames = 0
empty_chunks = 0

for chunk_dir in chunk_dirs:

    chunk_path = os.path.join(
        frames_dir,
        chunk_dir
    )

    # -----------------------------------------
    # PARSE CHUNK INDEX
    # -----------------------------------------

    try:
        chunk_index = int(
            chunk_dir.split("_")[-1]
        )
    except ValueError:
        chunk_index = 0

    # -----------------------------------------
    # CALCULATE TIME RANGE
    # -----------------------------------------

    start_sec = chunk_index * CHUNK_DURATION
    end_sec = start_sec + CHUNK_DURATION

    start_min = start_sec // 60
    start_s = start_sec % 60
    end_min = end_sec // 60
    end_s = end_sec % 60

    time_range = (
        f"{start_min}:{start_s:02d}"
        f" - "
        f"{end_min}:{end_s:02d}"
    )

    # -----------------------------------------
    # COLLECT FRAMES IN THIS CHUNK
    # -----------------------------------------

    frame_files = sorted([

        f for f in os.listdir(chunk_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ])

    # -----------------------------------------
    # HANDLE EMPTY CHUNKS
    # -----------------------------------------

    if not frame_files:

        print(
            f"  {chunk_dir}: No frames "
            f"(scene may be static)"
        )

        empty_chunks += 1

        chunks_data.append({
            "chunk_id": chunk_dir,
            "chunk_index": chunk_index,
            "time_range": time_range,
            "frames_analyzed": 0,
            "descriptions": [],
            "note": "No keyframes extracted."
        })

        continue

    # -----------------------------------------
    # PROCESS EACH FRAME
    # -----------------------------------------

    print(
        f"  {chunk_dir} [{time_range}]: "
        f"{len(frame_files)} frames"
    )

    frame_descriptions = []
    seen_in_chunk = set()

    for frame_file in frame_files:

        frame_path = os.path.join(
            chunk_path,
            frame_file
        )

        total_frames += 1

        try:

            image = Image.open(
                frame_path
            ).convert("RGB")

            # -----------------------------------------
            # RUN MOONDREAM
            # -----------------------------------------

            encoded = model.encode_image(image)

            answer = model.answer_question(
                encoded,
                VISION_PROMPT,
                tokenizer
            )

            description = answer.strip()

            # -----------------------------------------
            # SKIP EMPTY OR DUPLICATE
            # -----------------------------------------

            if not description:
                continue

            if description in seen_in_chunk:
                continue

            seen_in_chunk.add(description)

            frame_descriptions.append({
                "frame": frame_file,
                "description": description
            })

            print(f"    {frame_file}: OK")

        except Exception as e:

            failed_frames += 1

            print(f"    FAILED: {frame_file}: {e}")

    # -----------------------------------------
    # STORE CHUNK DATA
    # -----------------------------------------

    chunk_entry = {
        "chunk_id": chunk_dir,
        "chunk_index": chunk_index,
        "time_range": time_range,
        "frames_analyzed": len(frame_descriptions),
        "descriptions": frame_descriptions
    }

    if not frame_descriptions and frame_files:
        chunk_entry["note"] = (
            "Frames existed but all "
            "descriptions failed."
        )

    chunks_data.append(chunk_entry)

# =========================================
# 5. CLEANUP
# =========================================

print("\nFreeing Moondream2...")

del model
del tokenizer

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# =========================================
# 6. BUILD AND SAVE JSON
# =========================================

video_context = {

    "metadata": {

        "model": MODEL_ID,
        "revision": REVISION,
        "chunk_duration_sec": CHUNK_DURATION,
        "total_chunks": len(chunk_dirs),
        "total_frames_analyzed": total_frames,
        "failed_frames": failed_frames,
        "empty_chunks": empty_chunks,
        "processing_timestamp": (
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    },

    "chunks": chunks_data
}

with open(
    context_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        video_context,
        f,
        indent=4,
        ensure_ascii=False
    )

# =========================================
# 7. SUMMARY
# =========================================

print(f"\n{'=' * 40}")
print(f"Visual context saved: {context_output}")
print(f"Chunks processed: {len(chunk_dirs)}")
print(f"Frames analyzed:  {total_frames}")
print(f"Failed frames:    {failed_frames}")
print(f"Empty chunks:     {empty_chunks}")
print(f"{'=' * 40}\n")
