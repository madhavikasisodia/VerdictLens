"""Report legal entity coverage for extractive methods on a dev JSONL split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verdictlens.entities import extract_entities, extract_outcome_label
from verdictlens.extractive import EntityAwareExtractor, summarize_lexrank, summarize_textrank

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
METHODS = ("textrank", "lexrank", "entity_aware")


def load_dev_records(path: Path) -> list[dict[str, Any]]:
    """Load processed dev records containing doc_id and cleaned_sentences."""
    if not path.exists():
        raise FileNotFoundError(f"Dev split not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [dict(json.loads(line)) for line in handle if line.strip()]


def coverage(source: str, summary: str) -> dict[str, float]:
    """Calculate statute, party, and final-outcome coverage in a summary."""
    source_entities = extract_entities(source)
    summary_entities = extract_entities(summary)
    source_groups = {
        "statute": source_entities.statutes + source_entities.constitutional_articles,
        "party": source_entities.parties,
    }
    result: dict[str, float] = {}
    for name, values in source_groups.items():
        normalized_summary = summary.casefold()
        result[name] = sum(value.casefold() in normalized_summary for value in values) / len(values) if values else 1.0
    source_outcome = source_entities.outcome or extract_outcome_label(source, final_window=False)
    summary_outcome = summary_entities.outcome or extract_outcome_label(summary, final_window=False)
    result["outcome"] = 1.0 if not source_outcome or source_outcome == summary_outcome else 0.0
    return result


def summarize(method: str, text: str, word_budget: int, entity_extractor: EntityAwareExtractor) -> str:
    """Run one named extractive method."""
    if method == "textrank":
        return summarize_textrank(text, word_budget)
    if method == "lexrank":
        return summarize_lexrank(text, word_budget)
    return entity_extractor.summarize(text, word_budget)


def write_report(path: Path, totals: dict[str, dict[str, float]], count: int) -> None:
    """Write mean coverage metrics as markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Extractive Dev Coverage", "", f"Documents: {count}", "", "| Extractor | Statute coverage | Party coverage | Outcome coverage |", "| --- | ---: | ---: | ---: |"]
    for method in METHODS:
        metrics = totals[method]
        lines.append(f"| {method} | {metrics['statute'] / count:.3f} | {metrics['party'] / count:.3f} | {metrics['outcome'] / count:.3f} |" if count else f"| {method} | n/a | n/a | n/a |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extractive legal-entity coverage on dev")
    parser.add_argument("--input", type=Path, default=Path("data/splits/dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/extractive_coverage_dev.md"))
    parser.add_argument("--word-budget", type=int, default=3000)
    args = parser.parse_args()
    records = load_dev_records(args.input)
    extractor = EntityAwareExtractor()
    totals = {method: {"statute": 0.0, "party": 0.0, "outcome": 0.0} for method in METHODS}
    for record in records:
        text = " ".join(str(sentence) for sentence in record.get("cleaned_sentences", []))
        for method in METHODS:
            summary = summarize(method, text, args.word_budget, extractor)
            metrics = coverage(text, summary)
            for name, value in metrics.items():
                totals[method][name] += value
    write_report(args.output, totals, len(records))
    LOGGER.info("Evaluated %d dev documents; wrote %s", len(records), args.output)


if __name__ == "__main__":
    main()
