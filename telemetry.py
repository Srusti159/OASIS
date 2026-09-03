# =============================================================================
# telemetry.py
# Part 6 — Benchmarking Pipeline: Hardware Spy, JSON Schema Builder,
#           and Markdown Master Report Generator.
#
# Roles:
#   HardwareSpy       -> Background thread that tracks peak VRAM usage via NVML.
#   JSONSchemaBuilder -> Produces a standardised result skeleton for every run.
#   ReportGenerator   -> Aggregates all JSON results into a ranked Markdown report.
# =============================================================================

import os           # Directory / path utilities
import json         # Serialise and deserialise result files
import time         # sleep() in the monitor loop
import threading    # Daemon thread for background VRAM polling
import datetime     # ISO-8601 timestamp generation
import glob         # Wildcard scanning of evaluation JSON files

import psutil       # CPU / system memory introspection (available for future extension)
import pynvml       # NVIDIA Management Library Python bindings


# =============================================================================
# 1. HARDWARE SPY
# Polls GPU VRAM in a background daemon thread every 200 ms and records the
# peak allocation observed during a model's inference window.
# =============================================================================

class HardwareSpy:
    """
    Background VRAM tracker.

    Usage pattern:
        spy = HardwareSpy()
        spy.start()
        # ... run model inference ...
        peak_mb = spy.stop()
    """

    def __init__(self) -> None:
        # Initialise the NVML library — must be called before any NVML query.
        # Raises pynvml.NVMLError if no NVIDIA driver is present.
        pynvml.nvmlInit()

        # Peak VRAM seen so far (in MB); updated by the monitor thread.
        self.peak_vram_mb: float = 0.0

        # Guard flag read/written by both the main thread (stop) and the monitor thread.
        self.is_tracking: bool = False

        # Mutex protecting peak_vram_mb from concurrent read-write races.
        self._lock: threading.Lock = threading.Lock()

        # Placeholder for the background thread object; set in start().
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------------------
    def _monitor_loop(self) -> None:
        """
        Inner polling loop executed on the daemon thread.
        Queries VRAM every 200 ms and updates self.peak_vram_mb.
        Terminates when self.is_tracking flips to False (set by stop()).
        """
        # Obtain a handle to GPU device 0 (the primary A100).
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        while self.is_tracking:
            # nvmlDeviceGetMemoryInfo returns an object with .used in bytes.
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            current_vram_mb: float = mem_info.used / (1024 ** 2)   # bytes -> MB

            # Update peak under the lock to prevent a torn read in stop().
            with self._lock:
                if current_vram_mb > self.peak_vram_mb:
                    self.peak_vram_mb = current_vram_mb

            # Poll interval: 200 ms — fine-grained enough to catch short spikes.
            time.sleep(0.2)

    # -------------------------------------------------------------------------
    def start(self) -> None:
        """
        Arm the spy and launch the background monitor thread.
        The thread is a daemon so it will not block interpreter shutdown.
        """
        self.is_tracking = True

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="HardwareSpy-VRAMMonitor",
            daemon=True,          # Thread dies automatically if main process exits
        )
        self._thread.start()

    # -------------------------------------------------------------------------
    def stop(self) -> float:
        """
        Disarm the spy, join the monitor thread, shut down NVML, and return
        the exact peak VRAM observed (in MB) during the tracked window.

        Returns
        -------
        float
            Peak GPU VRAM allocation in megabytes.
        """
        # Signal the loop to exit on its next iteration check.
        self.is_tracking = False

        # Wait for the monitor thread to finish its current sleep and exit cleanly.
        if self._thread is not None:
            self._thread.join(timeout=2.0)   # 2 s grace period beyond the 0.2 s poll

        # Release the NVML library resources.
        pynvml.nvmlShutdown()

        # Return the peak VRAM recorded; read under lock for safety.
        with self._lock:
            return self.peak_vram_mb


# =============================================================================
# 2. JSON SCHEMA BUILDER
# Produces a fully-typed, standardised result skeleton for every
# (meeting, model) pair.  Null fields are filled in by later pipeline stages.
# =============================================================================

class JSONSchemaBuilder:
    """
    Factory for result JSON skeletons.

    All null fields are intended to be populated by downstream pipeline stages
    (transcription, OCR, inference, evaluation) before the file is written to disk.
    """

    @staticmethod
    def build_skeleton(meeting_id: str, model_name: str) -> dict:
        """
        Construct and return a standardised result dictionary.

        Parameters
        ----------
        meeting_id : str
            Unique identifier for the meeting being processed (e.g. "M1").
        model_name : str
            Short pipeline key for the LLM being benchmarked (e.g. "glm4").

        Returns
        -------
        dict
            A fully-typed skeleton with all fields initialised to null / defaults.
        """
        return {
            # ------------------------------------------------------------------
            # TOP-LEVEL EXPERIMENT METADATA
            # ------------------------------------------------------------------
            "experiment": meeting_id,               # Which meeting this result covers
            "dataset": "AMI Corpus",                # Fixed dataset label for this pipeline
            "temperature": 0.0,                     # Must match InferenceConfig.TEMPERATURE
            "timestamp": datetime.datetime.now(     # ISO-8601 wall-clock at result creation
                datetime.timezone.utc
            ).isoformat(),

            # ------------------------------------------------------------------
            # MODEL IDENTITY
            # Quantization, context length, and parameter count are filled in
            # dynamically by the loader stage once the model card is inspected.
            # ------------------------------------------------------------------
            "model": {
                "name": model_name,
                "parameter_size": None,             # e.g. "9B", "8B", "7B"
                "quantization": None,               # e.g. "bfloat16", "int4"
                "context_length": None,             # Max tokens the model supports
            },

            # ------------------------------------------------------------------
            # INPUT METADATA
            # Populated after the audio/PDF processing stages complete.
            # ------------------------------------------------------------------
            "input": {
                "meeting_id": meeting_id,
                "audio_duration_sec": None,         # Length of A1.wav in seconds
                "pdf_pages": None,                  # Number of pages in P1.pdf
                "num_speakers": None,               # Diarisation speaker count
                "input_tokens": None,               # Prompt token count fed to the LLM
                "output_tokens": None,              # Generated token count
                "input_mode": "multimodal",         # Fixed: audio transcript + slide OCR
            },

            # ------------------------------------------------------------------
            # EFFICIENCY METRICS
            # Populated by the inference harness after generation completes.
            # ------------------------------------------------------------------
            "efficiency": {
                "inference_time_sec": None,         # Wall time for model.generate()
                "latency_sec": None,                # Time-to-first-token (TTFT)
                "throughput_tokens_per_sec": None,  # output_tokens / inference_time_sec
                "peak_vram_mb": None,               # Peak VRAM from HardwareSpy.stop()
            },

            # ------------------------------------------------------------------
            # SUMMARY QUALITY METRICS
            # Populated by the evaluation stage (ROUGE, BERTScore, BARTScore).
            # ------------------------------------------------------------------
            "summary_evaluation": {
                "rouge": {
                    "rouge1": {"precision": None, "recall": None, "f1": None},
                    "rouge2": {"precision": None, "recall": None, "f1": None},
                    "rougeL": {"precision": None, "recall": None, "f1": None},
                },
                "bertscore": {
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "model": None,                  # e.g. "microsoft/deberta-xlarge-mnli"
                },
                "bartscore": {
                    "score": None,                  # Log probability (higher is better)
                    "model": None,                  # e.g. "facebook/bart-large-cnn"
                },
            },

            # ------------------------------------------------------------------
            # HALLUCINATION METRICS
            # Placeholder until a dedicated factual-consistency model is integrated.
            # ------------------------------------------------------------------
            "hallucination": {
                "factual_consistency_score": None,  # 0.0–1.0; model TBD
                "entity_hallucination_rate": None,  # % of entities not grounded in context
                "notes": (
                    "Placeholder metrics pending dedicated factual-consistency "
                    "model integration."
                ),
            },

            # ------------------------------------------------------------------
            # LLM-AS-JUDGE SCORE
            # Populated by the judge stage (Meta-Llama-3.1-8B-Instruct evaluator).
            # Typically a 0–10 scalar rating of overall summary quality.
            # ------------------------------------------------------------------
            "judge_score": None,
        }


# =============================================================================
# 3. REPORT GENERATOR
# Scans the evaluations directory, parses all JSON result files, computes a
# composite ranking score, and writes a formatted Markdown master report.
# =============================================================================

class ReportGenerator:
    """
    Aggregates per-model JSON results into a ranked Markdown comparison table.

    Composite Score formula
    -----------------------
    Each of the three main signals is min-max normalised across all models
    in the current set, then summed:

        composite = norm(ROUGE-L F1) + norm(BERTScore F1) + norm(Judge Score)

    Range: 0.0 – 3.0  (higher is better).
    Models with missing scores are ranked last.
    """

    # -------------------------------------------------------------------------
    @staticmethod
    def _safe_get(d: dict, *keys, default=None):
        """Traverse a nested dict safely; return default if any key is missing."""
        for key in keys:
            if not isinstance(d, dict):
                return default
            d = d.get(key, default)
        return d

    # -------------------------------------------------------------------------
    @staticmethod
    def _normalise(values: list[float | None]) -> list[float]:
        """
        Min-max normalise a list of values.
        None entries are mapped to 0.0 after normalisation so they rank last.
        """
        clean: list[float] = [v for v in values if v is not None]
        if not clean or max(clean) == min(clean):
            # All identical or all missing — give each a neutral 0.5 (or 0 if none)
            return [0.5 if v is not None else 0.0 for v in values]
        lo, hi = min(clean), max(clean)
        return [
            (v - lo) / (hi - lo) if v is not None else 0.0
            for v in values
        ]

    # -------------------------------------------------------------------------
    def generate_md(self, evals_dir: str, output_path: str) -> None:
        """
        Scan *evals_dir* for JSON result files, parse metrics, compute composite
        scores, and write a Markdown master report to *output_path*.

        Parameters
        ----------
        evals_dir : str
            Path to the directory containing R1_<model>.json files.
        output_path : str
            Destination path for the generated .md report.
        """
        # Collect all JSON files in the evaluations directory
        pattern: str = os.path.join(evals_dir, "*.json")
        json_files: list[str] = sorted(glob.glob(pattern))

        if not json_files:
            print(f"[ReportGenerator] No JSON files found in '{evals_dir}'. Aborting.")
            return

        # ------------------------------------------------------------------
        # PARSE METRICS FROM EACH FILE
        # ------------------------------------------------------------------
        rows: list[dict] = []

        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as fh:
                    data: dict = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[ReportGenerator] Skipping '{filepath}': {exc}")
                continue

            model_name: str = (
                ReportGenerator._safe_get(data, "model", "name", default="unknown")
            )

            # ROUGE-L F1
            rouge_l_f1: float | None = ReportGenerator._safe_get(
                data, "summary_evaluation", "rouge", "rougeL", "f1"
            )

            # BERTScore F1
            bert_f1: float | None = ReportGenerator._safe_get(
                data, "summary_evaluation", "bertscore", "f1"
            )

            # LLM-as-Judge Score (0–10 scale; normalised to 0–1 for composite)
            judge_raw: float | None = ReportGenerator._safe_get(data, "judge_score")
            judge_norm: float | None = (
                judge_raw / 10.0 if judge_raw is not None else None
            )

            # Peak VRAM (MB)
            peak_vram: float | None = ReportGenerator._safe_get(
                data, "efficiency", "peak_vram_mb"
            )

            # Throughput (tokens/sec)
            throughput: float | None = ReportGenerator._safe_get(
                data, "efficiency", "throughput_tokens_per_sec"
            )

            # Meeting / experiment ID
            meeting_id: str = ReportGenerator._safe_get(
                data, "experiment", default="N/A"
            )

            rows.append({
                "model":       model_name,
                "meeting_id":  meeting_id,
                "rouge_l_f1":  rouge_l_f1,
                "bert_f1":     bert_f1,
                "judge_raw":   judge_raw,        # original 0–10 for display
                "judge_norm":  judge_norm,       # 0–1 for composite
                "peak_vram":   peak_vram,
                "throughput":  throughput,
            })

        if not rows:
            print("[ReportGenerator] All files failed to parse. Aborting.")
            return

        # ------------------------------------------------------------------
        # COMPUTE COMPOSITE SCORES
        # norm(ROUGE-L F1) + norm(BERTScore F1) + norm(Judge / 10)
        # ------------------------------------------------------------------
        rouge_norm  = ReportGenerator._normalise([r["rouge_l_f1"] for r in rows])
        bert_norm   = ReportGenerator._normalise([r["bert_f1"]    for r in rows])
        judge_norm_ = ReportGenerator._normalise([r["judge_norm"] for r in rows])

        for i, row in enumerate(rows):
            row["composite"] = round(
                rouge_norm[i] + bert_norm[i] + judge_norm_[i], 4
            )

        # Sort descending by composite score — best model first
        rows.sort(key=lambda r: r["composite"], reverse=True)

        best: dict = rows[0]

        # ------------------------------------------------------------------
        # FORMAT HELPER
        # ------------------------------------------------------------------
        def _fmt(value, decimals: int = 4, suffix: str = "") -> str:
            """Return a formatted string or '—' for None."""
            if value is None:
                return "—"
            return f"{value:.{decimals}f}{suffix}"

        # ------------------------------------------------------------------
        # BUILD MARKDOWN CONTENT
        # ------------------------------------------------------------------
        report_ts: str = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        lines: list[str] = [
            "# 📊 Multimodal Summarisation — Benchmark Master Report",
            "",
            f"> **Generated:** {report_ts}  ",
            f"> **Dataset:** AMI Corpus  ",
            f"> **Meeting:** {rows[0]['meeting_id']}  ",
            f"> **Evaluation Scope:** ROUGE-L · BERTScore · LLM-as-Judge · VRAM · Throughput",
            "",
            "---",
            "",
            "## 🔬 Per-Model Metrics",
            "",
            # Table header — padded for readability
            "| Rank | Model | ROUGE-L F1 | BERTScore F1 | Judge (0–10) "
            "| Peak VRAM (MB) | Throughput (tok/s) | Composite Score |",
            "| :--: | :---- | ---------: | -----------: | :----------: "
            "| -------------: | -----------------: | --------------: |",
        ]

        # Table rows
        for rank, row in enumerate(rows, start=1):
            medal: str = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
            lines.append(
                f"| {medal} | **{row['model']}** "
                f"| {_fmt(row['rouge_l_f1'])} "
                f"| {_fmt(row['bert_f1'])} "
                f"| {_fmt(row['judge_raw'], decimals=1)} "
                f"| {_fmt(row['peak_vram'], decimals=1)} "
                f"| {_fmt(row['throughput'], decimals=2)} "
                f"| **{_fmt(row['composite'], decimals=4)}** |"
            )

        lines += [
            "",
            "---",
            "",
            "## 📐 Composite Score Methodology",
            "",
            "The **Composite Score** (range 0 – 3.0) is computed by:",
            "",
            "1. **Min-max normalising** ROUGE-L F1, BERTScore F1, and Judge Score "
            "   (scaled to 0–10 → 0–1) independently across all models in this run.",
            "2. **Summing** the three normalised values.",
            "",
            "```",
            "Composite = norm(ROUGE-L F1) + norm(BERTScore F1) + norm(Judge Score / 10)",
            "```",
            "",
            "> Models with **missing** metric values are assigned a normalised score "
            "of **0.0** for that dimension and ranked last.",
            "",
            "---",
            "",
            "## 🏆 Verdict",
            "",
            f"For the meeting **`{best['meeting_id']}`**, "
            f"the best-performing model is:",
            "",
            f"### ✅ `{best['model']}`",
            "",
            f"| Metric | Value |",
            f"| :----- | ----: |",
            f"| ROUGE-L F1           | {_fmt(best['rouge_l_f1'])} |",
            f"| BERTScore F1         | {_fmt(best['bert_f1'])} |",
            f"| Judge Score (0–10)   | {_fmt(best['judge_raw'], decimals=1)} |",
            f"| Peak VRAM (MB)       | {_fmt(best['peak_vram'], decimals=1)} |",
            f"| Throughput (tok/s)   | {_fmt(best['throughput'], decimals=2)} |",
            f"| **Composite Score**  | **{_fmt(best['composite'], decimals=4)}** |",
            "",
            f"**{best['model']}** achieved the highest composite score of "
            f"**{_fmt(best['composite'], decimals=4)} / 3.0**, indicating the strongest "
            "overall balance of lexical overlap, semantic similarity, and human-aligned "
            "quality for this specific audio/PDF input pair.",
            "",
            "---",
            "",
            "## 📁 Source Files",
            "",
            "| File | Description |",
            "| :--- | :---------- |",
        ]

        # List the source JSON files used
        for jf in json_files:
            lines.append(f"| `{os.path.basename(jf)}` | Raw evaluation result |")

        lines += [
            "",
            "---",
            "",
            "*Report auto-generated by `telemetry.ReportGenerator` — "
            "Multimodal Summarisation Benchmarking Pipeline.*",
        ]

        # ------------------------------------------------------------------
        # WRITE REPORT TO DISK
        # ------------------------------------------------------------------
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

        print(f"[ReportGenerator] Master report saved → '{output_path}'")