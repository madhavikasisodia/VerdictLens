from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Iterable

SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet", ".pdf"}
SUMMARY_NAMES = ("summary", "headnote", "synopsis", "abstract", "highlights", "short_summary")
TEXT_NAMES = ("text", "judgment", "document", "content", "body", "full_text")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML dataset configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs/data.yaml") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    return config


def configured_slug(config: dict[str, Any]) -> str:
    """Return the first configured Kaggle dataset slug."""
    slugs = config.get("dataset_slugs", [])
    if not isinstance(slugs, list) or not slugs or not isinstance(slugs[0], str):
        raise ValueError("configs/data.yaml must contain a non-empty dataset_slugs list")
    return slugs[0]


def download_dataset(slug: str) -> Path:
    """Download a Kaggle dataset explicitly through kagglehub."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub before downloading a dataset") from exc
    return Path(kagglehub.dataset_download(slug))


def discover_data_files(dataset_path: Path) -> list[Path]:
    """Find supported tabular or PDF files below a dataset directory."""
    if dataset_path.is_file():
        files = [dataset_path]
    else:
        files = [path for path in dataset_path.rglob("*") if path.is_file()]
    supported = sorted(path for path in files if path.suffix.casefold() in SUPPORTED_SUFFIXES)
    if not supported:
        raise FileNotFoundError(f"No CSV, JSON, JSONL, parquet, or PDF files found under {dataset_path}")
    return supported


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a supported tabular file into row mappings."""
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            return [dict(json.loads(line)) for line in handle if line.strip()]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("data", "records", "rows", "items"):
                if isinstance(payload.get(key), list):
                    return [dict(row) for row in payload[key] if isinstance(row, dict)]
        raise ValueError(f"JSON file {path} must contain a list of row objects")
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas and pyarrow to inspect parquet files") from exc
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def extract_pdf_records(paths: Iterable[Path], output_path: Path) -> list[dict[str, Any]]:
    """Extract judgment text from PDFs and persist normalized records as JSONL."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to inspect PDF judgments") from exc
    records: list[dict[str, Any]] = []
    failures = 0
    for path in paths:
        try:
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
            year = path.parent.name if path.parent.name.isdigit() else ""
            records.append({"id": path.stem, "source_path": str(path), "year": year, "text": text, "summary": ""})
        except Exception as exc:  # PDF parsers raise format-specific exceptions.
            failures += 1
            LOGGER.warning("Could not extract %s: %s", path, exc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    LOGGER.info("Extracted %d PDFs; %d failures; wrote %s", len(records), failures, output_path)
    return records


def choose_column(columns: Iterable[str], names: tuple[str, ...]) -> str | None:
    """Find a column by exact or substring name matching."""
    column_list = list(columns)
    lowered = {column.casefold(): column for column in column_list}
    for name in names:
        if name in lowered:
            return lowered[name]
    return next((column for column in column_list if any(name in column.casefold() for name in names)), None)


def summary_columns(columns: Iterable[str]) -> list[str]:
    """Return columns whose names suggest summaries or headnotes."""
    return [column for column in columns if any(name in column.casefold() for name in SUMMARY_NAMES)]


def word_lengths(records: list[dict[str, Any]], column: str | None) -> list[int]:
    """Return non-empty whitespace-token counts for one column."""
    if not column:
        return []
    return [len(str(row.get(column, "")).split()) for row in records if str(row.get(column, "")).strip()]


def format_stats(lengths: list[int]) -> str:
    """Format word-length statistics for console and markdown output."""
    if not lengths:
        return "no non-empty values"
    return f"min={min(lengths)}, max={max(lengths)}, mean={statistics.mean(lengths):.1f}, median={statistics.median(lengths):.1f}"


def inspect_file(path: Path) -> dict[str, Any]:
    """Inspect one supported tabular file and return report data."""
    records = read_records(path)
    return inspect_records(records, path)


def inspect_records(records: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    """Inspect normalized records and return report data."""
    columns = list(records[0].keys()) if records else []
    text_column = choose_column(columns, TEXT_NAMES)
    summaries = summary_columns(columns)
    summary_column = choose_column(columns, SUMMARY_NAMES)
    lengths = word_lengths(records, text_column)
    reference_count = sum(bool(str(row.get(summary_column, "")).strip()) for row in records) if summary_column else 0
    return {
        "path": path,
        "rows": len(records),
        "columns": columns,
        "text_column": text_column,
        "summary_columns": summaries,
        "word_lengths": lengths,
        "reference_count": reference_count,
    }


def write_report(report_path: Path, slug: str, inspections: list[dict[str, Any]]) -> None:
    """Write the dataset inspection report under outputs."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = sum(item["rows"] for item in inspections)
    reference_count = sum(item["reference_count"] for item in inspections)
    lines = [
        "# Dataset Inspection Report",
        "",
        f"- Kaggle dataset slug: `{slug}`",
        f"- Files inspected: {len(inspections)}",
        f"- Total rows: {total_rows}",
        f"- Rows with non-empty reference summaries: {reference_count}",
        "",
    ]
    for item in inspections:
        lines.extend([
            f"## `{item['path'].name}`",
            "",
            f"- Rows: {item['rows']}",
            f"- Columns: {', '.join(f'`{column}`' for column in item['columns']) or 'none'}",
            f"- Text column: `{item['text_column']}`" if item["text_column"] else "- Text column: not detected",
            f"- Word-length stats: {format_stats(item['word_lengths'])}",
            f"- Summary/headnote/synopsis-like columns: {', '.join(f'`{column}`' for column in item['summary_columns']) or 'none detected'}",
            f"- Rows with non-empty reference summaries: {item['reference_count']}",
            "",
        ])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect a configured Kaggle corpus")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--path", type=Path, help="Inspect a local dataset path instead of downloading")
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_report.md"))
    parser.add_argument("--extracted-output", type=Path, default=Path("outputs/extracted_judgments.jsonl"))
    args = parser.parse_args()
    config = load_config(args.config)
    slug = configured_slug(config)
    dataset_path = args.path or download_dataset(slug)
    data_files = discover_data_files(dataset_path)
    pdf_files = [path for path in data_files if path.suffix.casefold() == ".pdf"]
    if pdf_files:
        records = extract_pdf_records(pdf_files, args.extracted_output)
        inspections = [inspect_records(records, args.extracted_output)]
    else:
        inspections = [inspect_file(path) for path in data_files]
    write_report(args.output, slug, inspections)
    for item in inspections:
        LOGGER.info("%s: %d rows; columns=%s", item["path"], item["rows"], item["columns"])
        LOGGER.info("Word-length stats: %s", format_stats(item["word_lengths"]))
        LOGGER.info("Summary-like columns: %s", item["summary_columns"] or "none")
    LOGGER.info("Rows with non-empty reference summaries: %d", sum(item["reference_count"] for item in inspections))
    LOGGER.info("Wrote report to %s", args.output)


if __name__ == "__main__":
    main()
