"""
evaluation.py
=============

Dedicated accuracy-evaluation module for the meeting-summarization benchmarking
framework.

Responsibility (and ONLY this responsibility):
    Given a model-generated summary and a ground-truth reference summary,
    compute a comprehensive set of quality metrics (ROUGE-1/2/L, BERTScore,
    BARTScore, hallucination-rate placeholder) and assemble them into the
    canonical benchmark JSON schema used across the framework.

This module deliberately knows nothing about:
    - prompt construction
    - model inference / model loading for summarization
    - multimodal fusion
    - the run directory layout
`fusion_pipeline.py` (or `model_runner.py`) is responsible for orchestration:
it generates a summary, gathers input/experiment/efficiency metadata, and
then calls into this module to produce and persist the benchmark record.

Design notes for extensibility
-------------------------------
All metrics are registered in `METRIC_REGISTRY`. Each entry is a callable
with the signature:

    fn(generated: str, reference: str, **kwargs) -> dict

To add a new metric (BLEU, METEOR, BLEURT, COMET, an NLI-based factual
consistency score, semantic similarity, etc.) in the future:

    1. Write a `compute_<metric>(generated, reference, **kwargs) -> dict`
       function (see the existing `compute_rouge`, `compute_bertscore`,
       `compute_bartscore` as templates).
    2. Register it in `METRIC_REGISTRY` (and, if it belongs under
       `hallucination` instead of `summary_evaluation`, extend
       `compute_hallucination_metrics` similarly).
    3. Nothing else changes — `evaluate_summary()` and
       `build_benchmark_record()` automatically pick it up.

No architectural changes are required elsewhere in the pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy singletons for heavy models (BERTScore / BARTScore backbones).
# Loaded once per process, regardless of how many models are benchmarked in
# a run, to avoid re-loading a scoring model for every summarized model.
# ---------------------------------------------------------------------------

class _LazyModelCache:
    """Holds lazily-initialized, expensive scoring backends."""

    _bert_scorer = None
    _bart_scorer = None

    @classmethod
    def get_bert_scorer_kwargs(cls, lang: str = "en") -> dict:
        # bert_score.score() re-initializes cheaply if given a model type;
        # we just centralize the config here so it's consistent everywhere.
        return {
            "lang": lang,
            "model_type": "microsoft/deberta-xlarge-mnli",
            "rescale_with_baseline": True,
            "verbose": False,
        }

    @classmethod
    def get_bart_scorer(cls, device: str = "cpu", checkpoint: str = "facebook/bart-large-cnn"):
        if cls._bart_scorer is None:
            cls._bart_scorer = _BARTScorer(device=device, checkpoint=checkpoint)
        return cls._bart_scorer


class _BARTScorer:
    """
    Minimal BARTScore implementation (Yuan et al., 2021 — "BARTScore:
    Evaluating Generated Text as Text Generation").

    BARTScore treats evaluation as a generation task: it scores a candidate
    summary by the average log-likelihood that a pretrained BART model
    (conditioned on the reference) would generate that candidate — i.e. how
    "expected" the candidate is under the reference's distribution.

    We use the "reference -> candidate" direction (faithfulness-oriented),
    which is the direction the original BARTScore paper recommends for
    summarization evaluation.
    """

    def __init__(self, device: str = "cpu", checkpoint: str = "facebook/bart-large-cnn", max_length: int = 1024):
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer

        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()
        self.loss_fct = torch.nn.NLLLoss(
            reduction="none", ignore_index=self.model.config.pad_token_id
        )

    def score(self, srcs: list, tgts: list, batch_size: int = 4) -> list:
        import torch

        scores = []
        with torch.no_grad():
            for i in range(0, len(srcs), batch_size):
                src_batch = srcs[i : i + batch_size]
                tgt_batch = tgts[i : i + batch_size]

                encoded_src = self.tokenizer(
                    src_batch, max_length=self.max_length, truncation=True,
                    padding=True, return_tensors="pt",
                ).to(self.device)
                encoded_tgt = self.tokenizer(
                    tgt_batch, max_length=self.max_length, truncation=True,
                    padding=True, return_tensors="pt",
                ).to(self.device)

                output = self.model(
                    input_ids=encoded_src["input_ids"],
                    attention_mask=encoded_src["attention_mask"],
                    labels=encoded_tgt["input_ids"],
                )
                logits = output.logits.view(-1, self.model.config.vocab_size)
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                loss = self.loss_fct(log_probs, encoded_tgt["input_ids"].view(-1))
                loss = loss.view(encoded_tgt["input_ids"].shape[0], -1)
                token_counts = (encoded_tgt["input_ids"] != self.model.config.pad_token_id).sum(dim=1)
                per_seq_loss = loss.sum(dim=1) / token_counts.clamp(min=1)
                batch_scores = (-per_seq_loss).tolist()  # higher = better
                scores.extend(batch_scores)
        return scores


# ---------------------------------------------------------------------------
# Individual metric functions — each returns a small, self-contained dict.
# ---------------------------------------------------------------------------

def compute_rouge(generated: str, reference: str, **kwargs) -> Dict[str, Any]:
    """
    ROUGE-1, ROUGE-2, ROUGE-L, each with precision / recall / f1.
    Requires: pip install rouge_score
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        logger.warning("rouge_score not installed: %s", e)
        return _rouge_placeholder()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference, generated)

    return {
        "rouge1": {
            "precision": round(scores["rouge1"].precision, 4),
            "recall": round(scores["rouge1"].recall, 4),
            "f1": round(scores["rouge1"].fmeasure, 4),
        },
        "rouge2": {
            "precision": round(scores["rouge2"].precision, 4),
            "recall": round(scores["rouge2"].recall, 4),
            "f1": round(scores["rouge2"].fmeasure, 4),
        },
        "rougeL": {
            "precision": round(scores["rougeL"].precision, 4),
            "recall": round(scores["rougeL"].recall, 4),
            "f1": round(scores["rougeL"].fmeasure, 4),
        },
    }


def _rouge_placeholder() -> Dict[str, Any]:
    empty = {"precision": None, "recall": None, "f1": None}
    return {"rouge1": dict(empty), "rouge2": dict(empty), "rougeL": dict(empty)}


def compute_bertscore(generated: str, reference: str, lang: str = "en", **kwargs) -> Dict[str, Any]:
    """
    BERTScore precision / recall / F1 using a DeBERTa-xlarge-MNLI backbone
    (the strongest-correlating default per the BERTScore paper's leaderboard).
    Requires: pip install bert_score
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError as e:
        logger.warning("bert_score not installed: %s", e)
        return {"precision": None, "recall": None, "f1": None}

    kwargs_cfg = _LazyModelCache.get_bert_scorer_kwargs(lang=lang)
    P, R, F1 = bert_score_fn([generated], [reference], **kwargs_cfg)

    return {
        "precision": round(float(P.mean()), 4),
        "recall": round(float(R.mean()), 4),
        "f1": round(float(F1.mean()), 4),
    }


def compute_bartscore(
    generated: str,
    reference: str,
    device: str = "cpu",
    checkpoint: str = "facebook/bart-large-cnn",
    **kwargs,
) -> Dict[str, Any]:
    """
    BARTScore (reference -> candidate direction), suited for summarization
    faithfulness evaluation. Returns a single scalar log-likelihood score
    (higher / less negative = better).
    Requires: pip install torch transformers
    """
    try:
        scorer = _LazyModelCache.get_bart_scorer(device=device, checkpoint=checkpoint)
        score = scorer.score([reference], [generated])[0]
        return {"score": round(float(score), 4)}
    except Exception as e:  # noqa: BLE001 - want to degrade gracefully, not crash a benchmark run
        logger.warning("BARTScore computation failed: %s", e)
        return {"score": None}


def compute_hallucination_metrics(generated: str, reference: str, **kwargs) -> Dict[str, Any]:
    """
    Placeholder hallucination / factual-consistency section.

    `hallucination_rate` is currently a coarse lexical-overlap proxy
    (fraction of generated content words absent from the union of the
    reference + fused source context, if supplied via kwargs['source_text']).
    This is intentionally weak — it exists so the schema field is populated
    end-to-end — and is meant to be replaced by a proper NLI- or QA-based
    factual-consistency model (e.g. SummaC, QAFactEval, AlignScore) without
    changing the schema or the calling code.
    """
    source_text = kwargs.get("source_text", reference)
    hallucination_rate = _lexical_unsupported_fraction(generated, source_text)

    return {
        "hallucination_rate": hallucination_rate,
        # Placeholders for future factual-consistency metrics. Populate
        # these by registering new functions and merging their output here.
        "factual_consistency_score": None,   # e.g. SummaC / AlignScore
        "qa_based_consistency_score": None,  # e.g. QAFactEval
        "entailment_score": None,            # e.g. NLI-based entailment
        "notes": "Placeholder metrics pending dedicated factual-consistency model integration.",
    }


def _lexical_unsupported_fraction(generated: str, source_text: str) -> Optional[float]:
    """Very coarse proxy: fraction of unique generated words not present in source_text."""
    if not generated or not source_text:
        return None
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is",
        "are", "was", "were", "with", "as", "by", "that", "this", "it", "be",
        "at", "from", "will", "has", "have", "had", "their", "its",
    }
    gen_words = {w.strip(".,;:!?").lower() for w in generated.split()} - stopwords
    src_words = {w.strip(".,;:!?").lower() for w in source_text.split()} - stopwords
    if not gen_words:
        return None
    unsupported = gen_words - src_words
    return round(len(unsupported) / len(gen_words), 4)


# ---------------------------------------------------------------------------
# Metric registry — the single place new metrics get wired in.
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "rouge": compute_rouge,
    "bertscore": compute_bertscore,
    "bartscore": compute_bartscore,
    # Future entries, added without touching orchestration code:
    # "bleu": compute_bleu,
    # "meteor": compute_meteor,
    # "bleurt": compute_bleurt,
    # "comet": compute_comet,
}


def evaluate_summary(
    generated: str,
    reference: str,
    metrics: Optional[list] = None,
    metric_kwargs: Optional[Dict[str, dict]] = None,
) -> Dict[str, Any]:
    """
    Run the requested (default: all registered) metrics and return the
    `summary_evaluation` section of the benchmark schema.

    Parameters
    ----------
    generated : model-generated summary text
    reference : ground-truth reference summary text
    metrics   : subset of METRIC_REGISTRY keys to run; defaults to all
    metric_kwargs : optional per-metric kwargs, e.g.
                    {"bartscore": {"device": "cuda"}, "bertscore": {"lang": "en"}}
    """
    metrics = metrics or list(METRIC_REGISTRY.keys())
    metric_kwargs = metric_kwargs or {}

    results: Dict[str, Any] = {}
    for name in metrics:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            logger.warning("Unknown metric '%s' requested — skipping.", name)
            continue
        kwargs = metric_kwargs.get(name, {})
        try:
            results[name] = fn(generated, reference, **kwargs)
        except Exception as e:  # noqa: BLE001 - one failing metric shouldn't kill the whole eval
            logger.error("Metric '%s' failed: %s", name, e)
            results[name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Canonical benchmark schema assembly
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMeta:
    meeting_id: str
    dataset: str = "AMI Corpus"
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    run_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMeta:
    model_name: str
    parameter_size: Optional[str] = None      # e.g. "7B"
    quantization: Optional[str] = None        # e.g. "int4", "fp16", None
    context_length: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InputMeta:
    meeting_id: str
    audio_duration_sec: Optional[float] = None
    video_duration_sec: Optional[float] = None
    num_speakers: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    input_mode: str = "multimodal"  # "audio-only" or "multimodal"
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EfficiencyStats:
    inference_time_sec: Optional[float] = None
    latency_sec: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def build_benchmark_record(
    experiment: ExperimentMeta,
    model: ModelMeta,
    input_meta: InputMeta,
    generated_summary: str,
    reference_summary: str,
    efficiency: EfficiencyStats,
    metrics: Optional[list] = None,
    metric_kwargs: Optional[Dict[str, dict]] = None,
    source_text_for_hallucination: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assemble the full canonical benchmark JSON record for one model.

    This is the single function `fusion_pipeline.py` / `model_runner.py`
    should call once a model has produced a summary.
    """
    summary_evaluation = evaluate_summary(
        generated_summary, reference_summary, metrics=metrics, metric_kwargs=metric_kwargs
    )

    hallucination = compute_hallucination_metrics(
        generated_summary,
        reference_summary,
        source_text=source_text_for_hallucination or reference_summary,
    )

    record = {
        "experiment": {
            "meeting_id": experiment.meeting_id,
            "dataset": experiment.dataset,
            "temperature": experiment.temperature,
            "top_p": experiment.top_p,
            "top_k": experiment.top_k,
            "seed": experiment.seed,
            "run_id": experiment.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **experiment.extra,
        },
        "model": {
            "model_name": model.model_name,
            "parameter_size": model.parameter_size,
            "quantization": model.quantization,
            "context_length": model.context_length,
            **model.extra,
        },
        "input": {
            "meeting_id": input_meta.meeting_id,
            "audio_duration_sec": input_meta.audio_duration_sec,
            "video_duration_sec": input_meta.video_duration_sec,
            "num_speakers": input_meta.num_speakers,
            "input_tokens": input_meta.input_tokens,
            "output_tokens": input_meta.output_tokens,
            "input_mode": input_meta.input_mode,
            **input_meta.extra,
        },
        "summary_evaluation": summary_evaluation,
        "hallucination": hallucination,
        "efficiency": {
            "inference_time_sec": efficiency.inference_time_sec,
            "latency_sec": efficiency.latency_sec,
            "throughput_tokens_per_sec": efficiency.throughput_tokens_per_sec,
            "peak_vram_mb": efficiency.peak_vram_mb,
            **efficiency.extra,
        },
    }
    return record


def save_benchmark_record(record: Dict[str, Any], benchmarks_dir: Path, model_name: str) -> Path:
    """
    Persist the benchmark record as `benchmarks/<model_name>_benchmark.json`,
    the canonical benchmark artifact for that model in this run.
    """
    benchmarks_dir = Path(benchmarks_dir)
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace(" ", "_")
    output_path = benchmarks_dir / f"{safe_name}_benchmark.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    logger.info("Saved benchmark record: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Standalone CLI — mirrors the dual-mode (pipeline / standalone) pattern
# already used by fusion_pipeline.py, so evaluation.py can be run and
# tested independently of the rest of the pipeline.
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Standalone evaluation: compare a generated summary against a reference summary."
    )
    parser.add_argument("--generated", required=True, help="Path to <model_name>_summary.txt")
    parser.add_argument("--reference", required=True, help="Path to reference_summary.txt")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--meeting_id", default="standalone_run")
    parser.add_argument("--benchmarks_dir", default="./benchmarks")
    parser.add_argument("--device", default="cpu", help="Device for BERTScore/BARTScore")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generated_summary = _read_text(args.generated)
    reference_summary = _read_text(args.reference)

    record = build_benchmark_record(
        experiment=ExperimentMeta(meeting_id=args.meeting_id),
        model=ModelMeta(model_name=args.model_name),
        input_meta=InputMeta(meeting_id=args.meeting_id),
        generated_summary=generated_summary,
        reference_summary=reference_summary,
        efficiency=EfficiencyStats(),
        metric_kwargs={"bartscore": {"device": args.device}, "bertscore": {}},
    )

    save_benchmark_record(record, Path(args.benchmarks_dir), args.model_name)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()