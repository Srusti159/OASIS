"""
fusion_pipeline.py

Responsibilities (and only these):
- Load transcript.json + video_context.json
- Fuse them by timestamp overlap -> fused_context.json
  (or reuse an existing fused_context.json if one is already present)
- Build the multimodal prompt
- Invoke model_runner.run_all_models()
- For each model result: save its summary text, then hand off to
  evaluation.py to compute accuracy metrics and write the canonical
  benchmarks/<model_name>_benchmark.json

Metric computation itself (ROUGE / BERTScore / BARTScore / hallucination)
lives entirely in evaluation.py — this file only orchestrates.
"""

import os
import sys
import json
import math
from datetime import datetime

try:
    from model_runner import run_all_models
except Exception:
    run_all_models = None

try:
    from evaluation import (
        ExperimentMeta,
        ModelMeta,
        InputMeta,
        EfficiencyStats,
        build_benchmark_record,
        save_benchmark_record,
    )
except Exception:
    build_benchmark_record = None
    save_benchmark_record = None


def parse_time_range(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])

    if isinstance(value, str):
        value = value.replace("[", "").replace("]", "")
        value = value.replace(" ", "")
        if "-" in value:
            a, b = value.split("-")
        elif "," in value:
            a, b = value.split(",")
        else:
            return 0.0, 0.0

        def conv(x):
            if ":" in x:
                m, s = x.split(":")
                return int(m) * 60 + float(s)
            return float(x)

        return conv(a), conv(b)

    return 0.0, 0.0


def overlap(a1, a2, b1, b2):
    # Strict inequality: two ranges that only TOUCH at a boundary (e.g.
    # chunk 0-30 and chunk 30-60) are NOT overlapping. Using <= here would
    # make every fused chunk bleed its visual context into its neighbor,
    # since fuse()'s window size matches the video chunker's segment size
    # exactly (every chunk boundary touches the next chunk's start).
    return max(a1, b1) < min(a2, b2)


def fuse(transcript, vision, chunk_duration=30.0):
    """
    Groups visual context and audio segments into fixed duration chunks (default 30s).
    """
    vision_chunks = vision.get("chunks", [])
    transcript_segments = transcript.get("segments", [])

    # Find total duration to determine number of 30-second windows
    max_time = 0.0
    for chunk in vision_chunks:
        _, ve = parse_time_range(chunk.get("time_range", "0-0"))
        max_time = max(max_time, ve)

    for seg in transcript_segments:
        max_time = max(max_time, float(seg.get("end", 0.0)))

    total_chunks = math.ceil(max_time / chunk_duration) if max_time > 0 else 1
    fused_meeting = []

    for idx in range(total_chunks):
        c_start = idx * chunk_duration
        c_end = (idx + 1) * chunk_duration

        # 1. Collect video descriptions overlapping with this 30s window
        chunk_visuals = []
        for chunk in vision_chunks:
            vs, ve = parse_time_range(chunk.get("time_range", "0-0"))
            if overlap(c_start, c_end, vs, ve):
                for d in chunk.get("descriptions", []):
                    chunk_visuals.append(d.get("description", ""))

        # 2. Collect overlapping audio transcript segments nested inside
        nested_audio = []
        for seg in transcript_segments:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))

            if overlap(c_start, c_end, seg_start, seg_end):
                nested_audio.append({
                    "start": seg_start,
                    "end": seg_end,
                    "speaker": seg.get("speaker"),
                    "text": seg.get("text", "")
                })

        # Skip completely empty time windows
        if not chunk_visuals and not nested_audio:
            continue

        fused_meeting.append({
            "chunk_start": c_start,
            "chunk_end": c_end,
            "visual_context": chunk_visuals,
            "audio_segments": nested_audio
        })

    return {
        "metadata": {
            "created": datetime.now().isoformat(),
            "segments": len(fused_meeting),
            "chunk_duration_sec": chunk_duration
        },
        "meeting": fused_meeting
    }


def build_prompt(fused):
    lines = []

    for chunk in fused["meeting"]:
        audio_lines = []
        for a in chunk["audio_segments"]:
            speaker = a.get("speaker") or "Unknown Speaker"
            audio_lines.append(
                f"  [{a['start']:.1f}s - {a['end']:.1f}s] {speaker}: {a['text']}"
            )

        audio_str = "\n".join(audio_lines) if audio_lines else "  (No speech)"
        visual_str = "\n".join('- ' + v for v in chunk['visual_context']) if chunk['visual_context'] else '(No visual changes)'

        lines.append(
            f"""Time Window: {chunk['chunk_start']:.1f}s - {chunk['chunk_end']:.1f}s
Visual Context (Background/Action):
{visual_str}

Audio Transcript:
{audio_str}
---------------------"""
        )

    system_and_instructions = (
        "You are an expert multimodal meeting analyst.\n"
        "Analyze the chronological visual context and transcript provided below.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- DO NOT list or repeat raw visual frame descriptions (e.g., do not write 'A group of people are seated around a table').\n"
        "- SYNTHESIZE both visual actions (e.g., slides displayed, gestures, writing on board) and spoken discussion into a high-level narrative.\n"
        "- Focus on core concepts discussed, decisions made, and technical insights shared.\n\n"
        "Generate your response using strictly the following format:\n\n"
        "### 1. Executive Summary\n"
        "[Provide a concise 2-3 sentence overview of what occurred and what was accomplished in this meeting.]\n\n"
        "### 2. Main Discussion Points & Key Insights\n"
        "[Bullet points outlining the core topics, technical explanations, and agreements reached. Integrate visual cues like slides/whiteboard context where relevant.]\n\n"
        "### 3. Action Items & Decisions\n"
        "[List specific agreements, workflow steps, or next steps identified by speakers.]\n\n"
        "--- MULTIMODAL DATA ---\n\n"
    )

    return system_and_instructions + "\n".join(lines)


def build_source_text(fused):
    """
    Flattened transcript + visual context text across all 30s chunks.
    """
    parts = []
    for chunk in fused["meeting"]:
        parts.extend(chunk.get("visual_context", []))
        for audio in chunk.get("audio_segments", []):
            parts.append(audio.get("text", ""))
    return " ".join(p for p in parts if p)


def load_reference_summary(run_dir):
    reference_path = os.path.join(run_dir, "reference", "reference_summary.txt")
    if not os.path.isfile(reference_path):
        print(f"Warning: reference summary not found at {reference_path} — "
              f"accuracy metrics will be skipped.")
        return None
    with open(reference_path, encoding="utf8") as f:
        return f.read().strip()


def count_speakers(transcript):
    speakers = {seg.get("speaker") for seg in transcript.get("segments", [])}
    speakers.discard(None)
    return len(speakers) if speakers else None


def main():
    if len(sys.argv) == 2:
        run_dir = sys.argv[1]

        outputs = os.path.join(run_dir, "outputs")
        transcript_path = os.path.join(outputs, "transcript.json")
        vision_path = os.path.join(outputs, "video_context.json")
        fused_path = os.path.join(outputs, "fused_context.json")

    elif len(sys.argv) == 3:
        transcript_path = sys.argv[1]
        vision_path = sys.argv[2]

        outputs = os.path.dirname(os.path.abspath(transcript_path))
        fused_path = os.path.join(outputs, "fused_context.json")

        # standalone mode has no run_dir, so no reference summary / benchmark
        # metadata is available — fall back gracefully below.
        run_dir = None

    else:
        print("Usage:")
        print("python fusion_pipeline.py <run_dir>")
        print("or")
        print("python fusion_pipeline.py transcript.json video_context.json")
        sys.exit(1)

    with open(transcript_path, encoding="utf8") as f:
        transcript = json.load(f)

    with open(vision_path, encoding="utf8") as f:
        vision = json.load(f)

    # -----------------------------------------
    # REUSE existing fused_context.json if present
    # -----------------------------------------
    # Lets you re-run model inference + benchmarking (e.g. after tweaking
    # model_runner.py, adding a model, or fixing evaluation.py) WITHOUT
    # re-running fusion every time. Delete fused_context.json (or the whole
    # outputs/ folder) if you want it regenerated from scratch instead.
    if os.path.isfile(fused_path):
        print(f"Found existing fused context, reusing it: {fused_path}")
        with open(fused_path, encoding="utf8") as f:
            fused = json.load(f)
    else:
        fused = fuse(transcript, vision)

        with open(fused_path, "w", encoding="utf8") as f:
            json.dump(fused, f, indent=4, ensure_ascii=False)

        print("Saved:", fused_path)

    prompt = build_prompt(fused)

    if run_all_models is None:
        print("model_runner not available.")
        return

    # Move directory creation UP before model execution
    summaries_dir = os.path.join(outputs, "summaries")
    bench_dir = os.path.join(outputs, "benchmarks")
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(bench_dir, exist_ok=True)

    # -----------------------------------------
    # CHECK FOR EXISTING SUMMARIES
    # -----------------------------------------
    backend_keywords = ["qwen", "llama", "mistral", "gemma"]
    skip_models = []
    results = []

    existing_files = os.listdir(summaries_dir) if os.path.exists(summaries_dir) else []

    for backend in backend_keywords:
        # Matches files like 'Qwen2.5-7B_summary.txt', 'Llama-3-8B_summary.txt', 'Gemma-2-9B_summary.txt'
        matched_file = next(
            (f for f in existing_files if f.endswith(".txt") and backend in f.lower()),
            None
        )

        if matched_file:
            file_path = os.path.join(summaries_dir, matched_file)
            with open(file_path, "r", encoding="utf8") as f:
                summary_text = f.read().strip()

            if summary_text:
                # Clean name extracted for benchmarking (e.g. 'Qwen2.5-7B')
                model_name = matched_file.replace("_summary.txt", "").replace(".txt", "")
                
                print(f"Found existing summary '{matched_file}' -> Skipping {backend.upper()} inference.")
                
                skip_models.append(backend)
                results.append({
                    "model_name": model_name,
                    "summary": summary_text,
                    "latency_sec": None,
                    "tokens_per_second": None,
                    "peak_vram_mb": None
                })

    # Run only the models that don't have a summary yet
    new_results = run_all_models(prompt, skip_models=skip_models)
    results.extend(new_results)

    # -----------------------------------------
    # Metadata shared across all models in this run
    # -----------------------------------------
    reference_summary = load_reference_summary(run_dir) if run_dir else None
    source_text = build_source_text(fused)
    num_speakers = count_speakers(transcript)
    meeting_id = os.path.basename(run_dir.rstrip("/")) if run_dir else "standalone"

    for r in results:
        name = r["model_name"]

        with open(os.path.join(summaries_dir, f"{name}_summary.txt"), "w", encoding="utf8") as f:
            f.write(r["summary"])

        efficiency = EfficiencyStats(
            inference_time_sec=r.get("latency_sec"),
            latency_sec=r.get("latency_sec"),
            throughput_tokens_per_sec=r.get("tokens_per_second"),
            peak_vram_mb=r.get("peak_vram_mb"),
        ) if build_benchmark_record else None

        if build_benchmark_record is not None and reference_summary is not None:
            # Full canonical benchmark record, including accuracy metrics.
            record = build_benchmark_record(
                experiment=ExperimentMeta(meeting_id=meeting_id, run_id=meeting_id),
                model=ModelMeta(model_name=name),
                input_meta=InputMeta(
                    meeting_id=meeting_id,
                    num_speakers=num_speakers,
                    input_mode="multimodal",
                ),
                generated_summary=r["summary"],
                reference_summary=reference_summary,
                efficiency=efficiency,
                source_text_for_hallucination=source_text,
            )
            save_benchmark_record(record, bench_dir, name)
        else:
            # Fallback: evaluation.py unavailable or no reference summary
            # (e.g. standalone mode) — preserve the minimal efficiency-only
            # record so the pipeline still produces something usable.
            reason = "evaluation.py not importable" if build_benchmark_record is None else "no reference summary found"
            print(f"Skipping accuracy metrics for '{name}' ({reason}); writing efficiency-only benchmark.")

            bench = {
                "model_name": name,
                "latency_sec": r.get("latency_sec"),
                "tokens_per_second": r.get("tokens_per_second"),
                "peak_vram_mb": r.get("peak_vram_mb"),
                "summary_length": len(r.get("summary", "")),
                "timestamp": datetime.now().isoformat(),
                "note": reason,
            }

            os.makedirs(bench_dir, exist_ok=True)
            with open(os.path.join(bench_dir, f"{name}_benchmark.json"), "w") as f:
                json.dump(bench, f, indent=4)

    print("Fusion pipeline completed.")


if __name__ == "__main__":
    main()