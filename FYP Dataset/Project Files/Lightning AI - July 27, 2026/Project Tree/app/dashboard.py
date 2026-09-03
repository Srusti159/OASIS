import os
import glob
import json
import subprocess

import streamlit as st
import pandas as pd

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="Multimodal LLM Benchmark",
    layout="wide"
)

# -----------------------------------------
# TITLE
# -----------------------------------------

st.title(
    "Multimodal LLM Benchmarking System"
)

st.markdown(
    """
Upload a video and compare
multiple LLM summaries,
benchmarks, and performance.
"""
)

# -----------------------------------------
# VIDEO UPLOAD
# -----------------------------------------

uploaded_audio = st.file_uploader(
    "Upload Audio (.wav)",
    type=["wav"]
)

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4", "mov", "avi"]
)

# # -----------------------------------------
# # DIARIZATION OPTION
# # -----------------------------------------

# enable_diarization = st.checkbox(
#     "Enable Speaker Diarization",
#     value=False
# )

# -----------------------------------------
# PROCESS BUTTON
# -----------------------------------------

if uploaded_video is not None and uploaded_audio is not None:

    if st.button(
        "Run Benchmark Pipeline"
    ):

        # -----------------------------------------
        # SAVE VIDEO
        # -----------------------------------------

        uploads_dir = "temp_uploads"

        os.makedirs(
            uploads_dir,
            exist_ok=True
        )

        audio_path = os.path.join(
            uploads_dir,
            uploaded_audio.name
        )

        video_path = os.path.join(
            uploads_dir,
            uploaded_video.name
        )

        with open(
            audio_path,
            "wb"
        ) as f:

            f.write(
                uploaded_audio.read()
            )

        with open(
            video_path,
            "wb"
        ) as f:

            f.write(
                uploaded_video.read()
            )

        # -----------------------------------------
        # RUN PIPELINE
        # -----------------------------------------

        with st.spinner(
            "Processing video..."
        ):


            process = subprocess.run(

                [
                    "python",
                    "app/main.py",
                    audio_path,
                    video_path
                ],

                check = True
                # stdout=subprocess.DEVNULL,
                # stderr=subprocess.DEVNULL
            )

        # -----------------------------------------
        # FIND LATEST RUN
        # -----------------------------------------

        runs = sorted(
            glob.glob("runs/run_*")
        )

        latest_run = runs[-1]

        outputs_dir = os.path.join(
            latest_run,
            "outputs"
        )

        benchmark_dir = os.path.join(
            latest_run,
            "benchmarks"
        )

        plots_dir = os.path.join(
            latest_run,
            "plots"
        )

        # -----------------------------------------
        # SUCCESS
        # -----------------------------------------

        st.success(
            "Pipeline completed successfully!"
        )

        # -----------------------------------------
        # SUMMARIES
        # -----------------------------------------

        st.header(
            "LLM Summaries"
        )

        summary_files = glob.glob(
            os.path.join(
                outputs_dir,
                "*_summary.txt"
            )
        )

        for summary_file in summary_files:

            model_name = os.path.basename(
                summary_file
            ).replace(
                "_summary.txt",
                ""
            )

            with open(
                summary_file,
                "r",
                encoding="utf-8"
            ) as f:

                summary = f.read()

            with st.expander(
                model_name
            ):

                st.write(summary)

        # -----------------------------------------
        # BENCHMARK TABLE
        # -----------------------------------------

        st.header(
            "Benchmark Comparison"
        )

        benchmark_files = glob.glob(
            os.path.join(
                benchmark_dir,
                "*.json"
            )
        )

        benchmark_data = []

        for file in benchmark_files:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                benchmark_data.append(
                    json.load(f)
                )

        if benchmark_data:

            df = pd.DataFrame(
                benchmark_data
            )

            st.dataframe(
                df,
                use_container_width=True
            )

        # -----------------------------------------
        # PLOTS
        # -----------------------------------------

        st.header(
            "Benchmark Graphs"
        )

        plot_files = glob.glob(
            os.path.join(
                plots_dir,
                "*.png"
            )
        )

        for plot in plot_files:

            st.image(
                plot,
                use_container_width=True
            )

        # -----------------------------------------
        # RUN INFO
        # -----------------------------------------

        st.header(
            "Run Information"
        )

        st.write(
            f"Run Directory: {latest_run}"
        )