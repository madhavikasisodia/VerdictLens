"""Evaluation, confidence intervals, and paired comparisons."""

from __future__ import annotations

import csv
import json
import logging
import math
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .verify import verify_summary

LOGGER = logging.getLogger(__name__)

P7_NAMES = (
    "entity_precision",
    "statute_precision",
    "statute_recall",
    "outcome_match",
    "sentence_support",
    "foreign_clean",
    "faithful",
)


def token_f1(reference: str, prediction: str) -> float:
    """Compute unigram F1 without optional dependencies."""
    reference_tokens = reference.casefold().split()
    prediction_tokens = prediction.casefold().split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    row = [0] * (len(right) + 1)
    for left_token in left:
        previous = 0
        for index, right_token in enumerate(right, start=1):
            saved = row[index]
            row[index] = previous + 1 if left_token == right_token else max(row[index], row[index - 1])
            previous = saved
    return row[-1]


def _rouge_fallback(reference: str, prediction: str) -> dict[str, float]:
    """Compute simple F-measure ROUGE-1/2/L without optional packages."""
    def ngrams(text: str, size: int) -> Counter[tuple[str, ...]]:
        tokens = text.casefold().split()
        return Counter(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))

    result: dict[str, float] = {}
    for name, size in (("rouge1", 1), ("rouge2", 2)):
        reference_grams = ngrams(reference, size)
        prediction_grams = ngrams(prediction, size)
        overlap = sum((reference_grams & prediction_grams).values())
        result[name] = 2 * overlap / (sum(reference_grams.values()) + sum(prediction_grams.values())) if reference_grams and prediction_grams else 0.0
    reference_tokens = reference.casefold().split()
    prediction_tokens = prediction.casefold().split()
    lcs = _lcs_length(reference_tokens, prediction_tokens)
    result["rougeL"] = 2 * lcs / (len(reference_tokens) + len(prediction_tokens)) if reference_tokens and prediction_tokens else 0.0
    return result


def optional_rouge(reference: str, prediction: str) -> dict[str, float] | None:
    """Use rouge_score when installed, otherwise return the deterministic fallback."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return _rouge_fallback(reference, prediction)
    scores = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(reference, prediction)
    return {name: value.fmeasure for name, value in scores.items()}


def optional_bertscore(reference: str, prediction: str) -> dict[str, float] | None:
    """Compute BERTScore when the optional bert-score package is installed."""
    try:
        from bert_score import score
    except ImportError:
        return None
    precision, recall, f1 = score([prediction], [reference], lang="en", verbose=False)
    return {"bertscore_precision": float(precision[0]), "bertscore_recall": float(recall[0]), "bertscore_f1": float(f1[0])}


def p7_metrics(source: str, prediction: str, p3_outcome: str | None = None, check_support: bool = False) -> dict[str, float]:
    """Return seven faithfulness metrics derived from the verifier report."""
    report = verify_summary(source, prediction, p3_outcome=p3_outcome, check_support=check_support)
    support = float(not report.sentence_flags) if report.support_model_status not in {"support-model-unavailable", "sentence-transformers-unavailable"} else float("nan")
    return {
        "entity_precision": report.entity_precision,
        "statute_precision": report.statute_precision,
        "statute_recall": report.statute_recall,
        "outcome_match": float(report.outcome_match) if report.outcome_match is not None else float("nan"),
        "sentence_support": support,
        "foreign_clean": float(not report.foreign_leakage),
        "faithful": float(report.is_faithful),
    }


def evaluate_summary(source: str, reference: str, prediction: str) -> dict[str, Any]:
    """Return lightweight summary and faithfulness metrics."""
    report = verify_summary(source, prediction, check_support=False)
    return {"token_f1": token_f1(reference, prediction), "faithful": report.is_faithful, "unsupported_count": len(report.unsupported_entities), "missing_count": len(report.missing_entities)}


def bootstrap_mean_ci(values: Sequence[float], seed: int = 42, samples: int = 2000, confidence: float = 0.95) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap confidence interval, ignoring NaN."""
    clean = [float(value) for value in values if not math.isnan(float(value))]
    if not clean:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(clean, k=len(clean))) for _ in range(samples)]
    means.sort()
    alpha = (1 - confidence) / 2
    low_index = int(alpha * (samples - 1))
    high_index = int((1 - alpha) * (samples - 1))
    return statistics.mean(clean), means[low_index], means[high_index]


def paired_permutation_test(left: Sequence[float], right: Sequence[float], seed: int = 42, samples: int = 10000) -> dict[str, float]:
    """Return paired mean difference and a two-sided sign-flip p-value."""
    pairs = [(float(a), float(b)) for a, b in zip(left, right) if not math.isnan(float(a)) and not math.isnan(float(b))]
    if not pairs:
        return {"n": 0.0, "mean_difference": float("nan"), "p_value": float("nan")}
    differences = [a - b for a, b in pairs]
    observed = abs(statistics.mean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        randomized = [difference if rng.random() < 0.5 else -difference for difference in differences]
        if abs(statistics.mean(randomized)) >= observed:
            extreme += 1
    return {"n": float(len(pairs)), "mean_difference": statistics.mean(differences), "p_value": (extreme + 1) / (samples + 1)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records."""
    with path.open("r", encoding="utf-8") as handle:
        return [dict(json.loads(line)) for line in handle if line.strip()]


def prepare_weak_test_split(source_path: Path, test_path: Path, count: int = 200, seed: int = 42) -> int:
    """Create a deterministic test split from embedded HEADNOTE sections."""
    if not source_path.exists():
        raise FileNotFoundError(f"Test split is missing and source corpus was not found: {source_path}")
    records: list[dict[str, str]] = []
    for raw in load_jsonl(source_path):
        text = str(raw.get("text", ""))
        match = re.search(r"\bHEADNOTE\s*:\s*(.*?)(?=\n\s*JUDGMENT\b|\Z)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        summary = " ".join(re.sub(r"\s+", " ", line).strip() for line in match.group(1).splitlines() if line.strip())
        if summary:
            records.append({"doc_id": str(raw.get("id", len(records))), "text": text, "summary": summary})
    if len(records) < count:
        raise ValueError(f"Need {count} weakly supervised test records; found {len(records)}")
    random.Random(seed).shuffle(records)
    selected = records[:count]
    test_path.parent.mkdir(parents=True, exist_ok=True)
    with test_path.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return len(selected)


def evaluate_test_systems(test_path: Path, generation_dir: Path, output_dir: Path, expected_count: int = 200, check_support: bool = False, source_path: Path = Path("outputs/extracted_judgments.jsonl")) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate cached systems on the test set and write CSV/LaTeX tables."""
    if not test_path.exists():
        LOGGER.info("Preparing missing test split from embedded HEADNOTE sections")
        prepare_weak_test_split(source_path, test_path, expected_count)
    test_records = load_jsonl(test_path)
    if len(test_records) != expected_count:
        raise ValueError(f"Expected {expected_count} test judgments, found {len(test_records)}")
    by_id = {str(record.get("doc_id", record.get("id"))): record for record in test_records}
    caches = sorted(generation_dir.glob("*.jsonl"))
    if not caches:
        raise FileNotFoundError(f"No generation caches found under {generation_dir}")
    rows: list[dict[str, Any]] = []
    for cache in caches:
        predictions = load_jsonl(cache)
        system = cache.stem
        for prediction_record in predictions:
            doc_id = str(prediction_record.get("doc_id", ""))
            source_record = by_id.get(doc_id)
            if source_record is None:
                continue
            source = str(source_record.get("text", "")) or " ".join(source_record.get("cleaned_sentences", []))
            reference = next((str(source_record.get(key, "")).strip() for key in ("summary", "reference_summary", "target", "abstract") if str(source_record.get(key, "")).strip()), "")
            prediction = str(prediction_record.get("summary", ""))
            row = {"system": system, "doc_id": doc_id, "compression_ratio": len(prediction.split()) / max(len(source.split()), 1), "runtime_seconds": prediction_record.get("runtime_seconds", float("nan"))}
            row.update({f"p7_{key}": value for key, value in p7_metrics(source, prediction, check_support=check_support).items()})
            if reference.strip():
                row.update(optional_rouge(reference, prediction) or {})
                row.update(optional_bertscore(reference, prediction) or {})
            rows.append(row)
    summary_rows = summarize_rows(rows)
    significance_rows = significance_rows_for(summary_rows, rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_metrics.csv", summary_rows)
    write_latex(output_dir / "evaluation_metrics.tex", summary_rows, "Evaluation metrics")
    write_csv(output_dir / "paired_significance.csv", significance_rows)
    write_latex(output_dir / "paired_significance.tex", significance_rows, "Paired significance")
    return summary_rows, significance_rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-document rows into mean and bootstrap CI rows."""
    systems = sorted({str(row["system"]) for row in rows})
    metrics = sorted(key for key in rows[0] if key not in {"system", "doc_id"}) if rows else []
    output: list[dict[str, Any]] = []
    for system in systems:
        system_rows = [row for row in rows if row["system"] == system]
        result: dict[str, Any] = {"system": system, "n": len(system_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in system_rows if isinstance(row.get(metric), (float, int))]
            mean, low, high = bootstrap_mean_ci(values)
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci_low"] = low
            result[f"{metric}_ci_high"] = high
        output.append(result)
    return output


def significance_rows_for(summary_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Run paired A4-vs-A3 and A3-vs-A2 tests for every numeric metric."""
    metrics = sorted(key for key in rows[0] if key not in {"system", "doc_id"}) if rows else []
    aliases = {
        "A2": {"A2", "pegasus_truncated"},
        "A3": {"A3", "pegasus_chunked"},
        "A4": {"A4", "entity_aware", "agent_repaired"},
    }
    by_system = {
        label: {str(row["doc_id"]): row for row in rows if str(row["system"]) in aliases[label]}
        for label in ("A2", "A3", "A4")
    }
    output: list[dict[str, Any]] = []
    for left, right in (("A4", "A3"), ("A3", "A2")):
        common_ids = sorted(set(by_system.get(left, {})) & set(by_system.get(right, {})))
        for metric in metrics:
            result = paired_permutation_test([by_system[left][doc_id][metric] for doc_id in common_ids], [by_system[right][doc_id][metric] for doc_id in common_ids]) if common_ids else {"n": 0.0, "mean_difference": float("nan"), "p_value": float("nan")}
            output.append({"comparison": f"{left} vs {right}", "metric": metric, **result})
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_latex(path: Path, rows: Sequence[Mapping[str, Any]], caption: str) -> None:
    """Write a compact LaTeX table without requiring pandas."""
    fields = list(rows[0].keys()) if rows else []
    line_break = chr(92) * 2
    lines = ["% " + caption, r"\begin{tabular}{" + "l" * len(fields) + "}", r"\toprule"]
    lines.append(" & ".join(fields) + " " + line_break)
    lines.append(r"\midrule")
    lines.extend(" & ".join(str(row.get(field, "")) for field in fields) + " " + line_break for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
