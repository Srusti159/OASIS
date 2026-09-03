import os
import sys
import shutil
import subprocess

from datetime import datetime

# -----------------------------------------
# INPUT VALIDATION
# -----------------------------------------

if len(sys.argv) != 4:

    print("Usage:")
    print("python app/main.py <audio.wav> <video.avi/.mp4> <reference_summary.txt>")

    sys.exit(1)

audio_input = sys.argv[1]
video_input = sys.argv[2]
reference_input = sys.argv[3]

# -----------------------------------------
# VALIDATE FILES
# -----------------------------------------

if not os.path.isfile(audio_input):
    print("Audio file not found!")
    sys.exit(1)

if not os.path.isfile(video_input):
    print("Video file not found!")
    sys.exit(1)

if not os.path.isfile(reference_input):
    print("Reference summary file not found!")
    sys.exit(1)

# -----------------------------------------
# VALIDATE EXTENSIONS
# -----------------------------------------

audio_ext = os.path.splitext(audio_input)[1].lower()
video_ext = os.path.splitext(video_input)[1].lower()
reference_ext = os.path.splitext(reference_input)[1].lower()

if audio_ext != ".wav":
    print("Audio must be a .wav file.")
    sys.exit(1)

if video_ext not in [".avi", ".mp4"]:
    print("Video must be .avi or .mp4.")
    sys.exit(1)

if reference_ext != ".txt":
    print("Reference summary must be a .txt file.")
    sys.exit(1)

# -----------------------------------------
# CREATE RUN DIRECTORY
# -----------------------------------------

run_id = datetime.now().strftime(
    "run_%Y%m%d_%H%M%S"
)

run_dir = os.path.join(
    "runs",
    run_id
)

audio_dir = os.path.join(run_dir, "audio")
video_dir = os.path.join(run_dir, "video")
reference_dir = os.path.join(run_dir, "reference")
outputs_dir = os.path.join(run_dir, "outputs")

os.makedirs(audio_dir, exist_ok=True)
os.makedirs(video_dir, exist_ok=True)
os.makedirs(reference_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

print(f"\nRun ID: {run_id}")
print(f"Run Directory: {run_dir}")

# -----------------------------------------
# COPY INPUT FILES
# -----------------------------------------

audio_path = os.path.join(audio_dir, "audio.wav")

video_filename = os.path.basename(video_input)
video_path = os.path.join(video_dir, video_filename)

reference_path = os.path.join(
    reference_dir,
    "reference_summary.txt"
)

shutil.copy2(audio_input, audio_path)
shutil.copy2(video_input, video_path)
shutil.copy2(reference_input, reference_path)

print("\nInput files copied successfully.")

# -----------------------------------------
# PREPROCESSING
# -----------------------------------------
# preprocess.py expects three positional args, in this order:
#   sys.argv[1] -> AUDIO_PATH
#   sys.argv[2] -> VIDEO_PATH
#   sys.argv[3] -> RUN_DIR
# (NOT just run_dir). preprocess.py does its OWN internal copy of the audio
# file into run_dir/audio/audio.wav, so it must be given the ORIGINAL
# (pre-copy) input paths here — passing the already-copied audio_path would
# make it copy that file onto itself (shutil.SameFileError).

print("\nRunning preprocessing...\n")

subprocess.run(
    [
        "python",
        "app/preprocess.py",
        audio_input,
        video_input,
        run_dir
    ], check = True
)

# -----------------------------------------
# TRANSCRIPTION
# -----------------------------------------

print("\nStarting transcription...\n")

subprocess.run(
    [
        "python",
        "app/transcribe.py",
        run_dir
    ],
    check=True
)

# -----------------------------------------
# VIDEO CONTEXT
# -----------------------------------------

print("\nGenerating video context...\n")

subprocess.run(
    [
        "python",
        "app/moondream_context.py",
        run_dir
    ],
    check=True
)

# -----------------------------------------
# MULTIMODAL FUSION
# -----------------------------------------

print("\nRunning fusion pipeline...\n")

subprocess.run(
    [
        "python",
        "app/fusion_pipeline.py",
        run_dir
    ],
    check=True
)

# -----------------------------------------
# BENCHMARKING
# -----------------------------------------

# print("\nRunning benchmark...\n")

# subprocess.run(
#     [
#         "python",
#         "app/benchmark.py",
#         run_dir
#     ],
#     check=True
# )

# -----------------------------------------
# VISUALIZATION
# -----------------------------------------

print("\nGenerating benchmark visualizations...\n")

subprocess.run(
    [
        "python",
        "app/visualize_benchmarks.py",
        run_dir
    ],
    check=True
)

# -----------------------------------------
# COMPLETE
# -----------------------------------------

print("\nPipeline completed successfully!")

print(f"Results stored in: {run_dir}")
