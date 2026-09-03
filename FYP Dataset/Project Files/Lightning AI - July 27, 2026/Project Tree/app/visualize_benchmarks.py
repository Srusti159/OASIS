import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------
# INPUT
# -----------------------------------------

RUN_DIR = sys.argv[1]

# -----------------------------------------
# PATHS
# -----------------------------------------
# Benchmarks now live inside outputs/ (see fusion_pipeline.py):
#   RUN_DIR/outputs/benchmarks/<model_name>_benchmark.json
# NOT RUN_DIR/benchmarks/ (that was the old, pre-reorg location).

benchmark_dir = os.path.join(
    RUN_DIR,
    "outputs",
    "benchmarks"
)

plots_dir = os.path.join(
    RUN_DIR,
    "plots"
)

os.makedirs(
    plots_dir,
    exist_ok=True
)

# -----------------------------------------
# LOAD BENCHMARK FILES
# -----------------------------------------

if not os.path.isdir(benchmark_dir):
    print(f"\nBenchmark directory not found: {benchmark_dir}")
    sys.exit(1)

benchmark_files = [

    os.path.join(
        benchmark_dir,
        file
    )

    for file in os.listdir(
        benchmark_dir
    )

    if file.endswith(".json")
]

# -----------------------------------------
# CHECK FILES
# -----------------------------------------

if not benchmark_files:

    print(
        "\nNo benchmark files found!"
    )

    sys.exit(1)

# -----------------------------------------
# LOAD JSON DATA
# -----------------------------------------

raw_records = []

for file in benchmark_files:

    try:

        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            benchmark = json.load(f)

            raw_records.append(
                benchmark
            )

    except Exception as e:

        print(
            f"Failed to load {file}: {e}"
        )

# -----------------------------------------
# CHECK DATA
# -----------------------------------------

if not raw_records:

    print(
        "\nNo valid benchmark data found!"
    )

    sys.exit(1)


# -----------------------------------------
# FLATTEN RECORDS
# -----------------------------------------
# Two possible shapes land here:
#   1. Canonical schema (evaluation.py):
#        {"model": {"model_name": ...}, "efficiency": {"latency_sec": ...},
#         "summary_evaluation": {"rouge": {...}, "bertscore": {...},
#         "bartscore": {...}}, "hallucination": {...}, ...}
#   2. Flat fallback schema (fusion_pipeline.py, when evaluation.py or a
#      reference summary wasn't available):
#        {"model_name": ..., "latency_sec": ..., "tokens_per_second": ...,
#         "peak_vram_mb": ..., "summary_length": ..., "note": ...}
# flatten_record() normalizes either into one flat row so downstream
# plotting code doesn't need to care which one produced a given file.

def _get(d, *keys, default=None):
    """Safe nested dict getter: _get(d, 'a', 'b') -> d['a']['b'] or default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_record(record):
    is_canonical = "model" in record and isinstance(record.get("model"), dict)

    if is_canonical:
        row = {
            "model_name": _get(record, "model", "model_name"),
            "latency_sec": _get(record, "efficiency", "latency_sec"),
            "tokens_per_second": _get(record, "efficiency", "throughput_tokens_per_sec"),
            "peak_vram_mb": _get(record, "efficiency", "peak_vram_mb"),
            "output_tokens": _get(record, "input", "output_tokens"),
            "rouge1_f1": _get(record, "summary_evaluation", "rouge", "rouge1", "f1"),
            "rouge2_f1": _get(record, "summary_evaluation", "rouge", "rouge2", "f1"),
            "rougeL_f1": _get(record, "summary_evaluation", "rouge", "rougeL", "f1"),
            "bertscore_f1": _get(record, "summary_evaluation", "bertscore", "f1"),
            "bartscore_score": _get(record, "summary_evaluation", "bartscore", "score"),
            "hallucination_rate": _get(record, "hallucination", "hallucination_rate"),
            "schema": "canonical",
        }
    else:
        # Flat fallback record — accuracy metrics simply weren't computed.
        row = {
            "model_name": record.get("model_name"),
            "latency_sec": record.get("latency_sec"),
            "tokens_per_second": record.get("tokens_per_second"),
            "peak_vram_mb": record.get("peak_vram_mb"),
            "output_tokens": None,
            "rouge1_f1": None,
            "rouge2_f1": None,
            "rougeL_f1": None,
            "bertscore_f1": None,
            "bartscore_score": None,
            "hallucination_rate": None,
            "schema": "fallback",
        }

    return row


data = [flatten_record(r) for r in raw_records]

# -----------------------------------------
# CREATE DATAFRAME
# -----------------------------------------

df = pd.DataFrame(data)

if df["model_name"].isnull().any():
    print("\nWarning: one or more benchmark files had no resolvable model_name — check the raw JSON.")

fallback_models = df.loc[df["schema"] == "fallback", "model_name"].tolist()
if fallback_models:
    print(f"\nNote: accuracy metrics unavailable for: {', '.join(fallback_models)} "
          f"(fallback/efficiency-only benchmark record).")

print("\n=========== BENCHMARK DATA ===========\n")

print(df)

print("\n======================================\n")


# -----------------------------------------
# PLOT HELPER
# -----------------------------------------

def bar_plot(column, ylabel, title, filename):
    """Skip cleanly if every value for this metric is missing (e.g. all
    models fell back to the efficiency-only schema)."""
    if column not in df or df[column].dropna().empty:
        print(f"Skipping '{title}' — no data available for '{column}'.")
        return

    plot_df = df.dropna(subset=[column])

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["model_name"], plot_df[column])
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, filename))
    plt.close()


# -----------------------------------------
# EFFICIENCY GRAPHS
# -----------------------------------------

bar_plot("latency_sec", "Latency (sec)", "LLM Inference Latency", "latency_comparison.png")
bar_plot("tokens_per_second", "Tokens/sec", "LLM Throughput", "throughput_comparison.png")
bar_plot("peak_vram_mb", "VRAM (MB)", "Peak VRAM Usage", "vram_comparison.png")
bar_plot("output_tokens", "Output Tokens", "Summary Length (Output Tokens)", "summary_length_comparison.png")

# -----------------------------------------
# ACCURACY GRAPHS
# -----------------------------------------

bar_plot("rougeL_f1", "ROUGE-L F1", "ROUGE-L F1 Comparison", "rougeL_comparison.png")
bar_plot("bertscore_f1", "BERTScore F1", "BERTScore F1 Comparison", "bertscore_comparison.png")
bar_plot("bartscore_score", "BARTScore", "BARTScore Comparison", "bartscore_comparison.png")
bar_plot("hallucination_rate", "Hallucination Rate", "Hallucination Rate Comparison", "hallucination_comparison.png")

# -----------------------------------------
# COMPLETE
# -----------------------------------------

print(
    "\nBenchmark graphs generated successfully!"
)



# import os
# import sys
# import json
# import pandas as pd
# import matplotlib.pyplot as plt

# # -----------------------------------------
# # INPUT
# # -----------------------------------------

# RUN_DIR = sys.argv[1]

# # -----------------------------------------
# # PATHS
# # -----------------------------------------

# benchmark_dir = os.path.join(
#     RUN_DIR,
#     "benchmarks"
# )

# plots_dir = os.path.join(
#     RUN_DIR,
#     "plots"
# )

# os.makedirs(
#     plots_dir,
#     exist_ok=True
# )

# # -----------------------------------------
# # LOAD BENCHMARK FILES
# # -----------------------------------------

# benchmark_files = [

#     os.path.join(
#         benchmark_dir,
#         file
#     )

#     for file in os.listdir(
#         benchmark_dir
#     )

#     if file.endswith(".json")
# ]

# # -----------------------------------------
# # CHECK FILES
# # -----------------------------------------

# if not benchmark_files:

#     print(
#         "\nNo benchmark files found!"
#     )

#     sys.exit(1)

# # -----------------------------------------
# # LOAD JSON DATA
# # -----------------------------------------

# data = []

# for file in benchmark_files:

#     try:

#         with open(
#             file,
#             "r",
#             encoding="utf-8"
#         ) as f:

#             benchmark = json.load(f)

#             data.append(
#                 benchmark
#             )

#     except Exception as e:

#         print(
#             f"Failed to load {file}: {e}"
#         )

# # -----------------------------------------
# # CHECK DATA
# # -----------------------------------------

# if not data:

#     print(
#         "\nNo valid benchmark data found!"
#     )

#     sys.exit(1)

# # -----------------------------------------
# # CREATE DATAFRAME
# # -----------------------------------------

# df = pd.DataFrame(data)

# print("\n=========== BENCHMARK DATA ===========\n")

# print(df)

# print("\n======================================\n")

# # -----------------------------------------
# # GRAPH 1 — LATENCY
# # -----------------------------------------

# plt.figure(figsize=(10, 5))

# plt.bar(
#     df["model_name"],
#     df["latency_sec"]
# )

# plt.xlabel("Model")

# plt.ylabel("Latency (sec)")

# plt.title(
#     "LLM Inference Latency"
# )

# plt.xticks(rotation=15)

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         plots_dir,
#         "latency_comparison.png"
#     )
# )

# plt.close()

# # -----------------------------------------
# # GRAPH 2 — TOKENS PER SECOND
# # -----------------------------------------

# plt.figure(figsize=(10, 5))

# plt.bar(
#     df["model_name"],
#     df["tokens_per_second"]
# )

# plt.xlabel("Model")

# plt.ylabel("Tokens/sec")

# plt.title(
#     "LLM Throughput"
# )

# plt.xticks(rotation=15)

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         plots_dir,
#         "throughput_comparison.png"
#     )
# )

# plt.close()

# # -----------------------------------------
# # GRAPH 3 — VRAM USAGE
# # -----------------------------------------

# plt.figure(figsize=(10, 5))

# plt.bar(
#     df["model_name"],
#     df["peak_vram_mb"]
# )

# plt.xlabel("Model")

# plt.ylabel("VRAM (MB)")

# plt.title(
#     "Peak VRAM Usage"
# )

# plt.xticks(rotation=15)

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         plots_dir,
#         "vram_comparison.png"
#     )
# )

# plt.close()

# # -----------------------------------------
# # GRAPH 4 — SUMMARY LENGTH
# # -----------------------------------------

# plt.figure(figsize=(10, 5))

# plt.bar(
#     df["model_name"],
#     df["summary_length"]
# )

# plt.xlabel("Model")

# plt.ylabel("Characters")

# plt.title(
#     "Summary Length Comparison"
# )

# plt.xticks(rotation=15)

# plt.tight_layout()

# plt.savefig(
#     os.path.join(
#         plots_dir,
#         "summary_length_comparison.png"
#     )
# )

# plt.close()

# # -----------------------------------------
# # COMPLETE
# # -----------------------------------------

# print(
#     "\nBenchmark graphs generated successfully!"
# )