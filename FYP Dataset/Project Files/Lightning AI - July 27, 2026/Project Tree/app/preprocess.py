import os
import sys
import subprocess
import shutil

# -----------------------------------------
# INPUTS
# -----------------------------------------

AUDIO_PATH = sys.argv[1]
VIDEO_PATH = sys.argv[2]

RUN_DIR = sys.argv[3]

# -----------------------------------------
# CREATE DIRECTORIES
# -----------------------------------------

audio_dir = os.path.join(
    RUN_DIR,
    "audio"
)

chunks_dir = os.path.join(
    RUN_DIR,
    "chunks"
)

frames_dir = os.path.join(
    RUN_DIR,
    "frames"
)

os.makedirs(
    audio_dir,
    exist_ok=True
)

os.makedirs(
    chunks_dir,
    exist_ok=True
)

os.makedirs(
    frames_dir,
    exist_ok=True
)

# -----------------------------------------
# AUDIO EXTRACTION
# -----------------------------------------

audio_output = os.path.join(
    audio_dir,
    "audio.wav"
)

shutil.copy2(
    AUDIO_PATH,
    audio_output
)

print("Audio copied.")

# print(
#     "Extracting audio..."
# )

# audio_command = [

#     "ffmpeg",

#     "-loglevel",
#     "quiet",

#     "-i",
#     VIDEO_PATH,

#     "-q:a",
#     "0",

#     "-map",
#     "a",

#     audio_output,

#     "-y"
# ]

# subprocess.run(

#     audio_command,

#     stdout=subprocess.DEVNULL,
#     stderr=subprocess.DEVNULL
# )

# print(
#     "Audio extraction completed"
# )

# -----------------------------------------
# VIDEO CHUNKING
# -----------------------------------------

print(
    "Splitting video into chunks..."
)

chunk_output = os.path.join(
    chunks_dir,
    "chunk_%03d.mp4"
)

chunk_command = [

    "ffmpeg",

    "-loglevel",
    "quiet",

    "-i",
    VIDEO_PATH,

    "-c",
    "copy",

    "-map",
    "0",

    "-segment_time",
    "30",

    "-f",
    "segment",

    "-reset_timestamps",
    "1",

    chunk_output,

    "-y"
]

subprocess.run(

    chunk_command,

    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

print(
    "Video chunking completed"
)

# -----------------------------------------
# KEYFRAME EXTRACTION
# -----------------------------------------

print(
    "Extracting keyframes..."
)

chunk_files = sorted([

    f for f in os.listdir(
        chunks_dir
    )

    if f.endswith(".mp4")
])

for chunk in chunk_files:

    chunk_name = chunk.replace(
        ".mp4",
        ""
    )

    chunk_path = os.path.join(
        chunks_dir,
        chunk
    )

    chunk_frame_dir = os.path.join(
        frames_dir,
        chunk_name
    )

    os.makedirs(
        chunk_frame_dir,
        exist_ok=True
    )

    frame_output = os.path.join(
        chunk_frame_dir,
        "frame_%03d.jpg"
    )

    # -----------------------------------------
    # SCENE-BASED KEYFRAMES
    # -----------------------------------------
    frame_command = [

        "ffmpeg",

        "-loglevel",
        "quiet",

        "-i",
        chunk_path,

        "-vf",
        "fps=1/10",

        frame_output,

        "-y"
    ]

    subprocess.run(

        frame_command,

        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

print(
    "Keyframe extraction completed"
)

# -----------------------------------------
# COMPLETE
# -----------------------------------------

print(
    "\nPreprocessing completed successfully"
)