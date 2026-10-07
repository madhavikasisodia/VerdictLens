"""Text and tabular corpus preprocessing utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)
MIN_JUDGMENT_WORDS = 1_500
MAX_JUDGMENT_WORDS = 60_000
FIXED_SPLIT_SIZES = {"train": 2_000, "dev": 100, "test": 200}


def remove_boilerplate(text: str | None) -> str:
    """Remove common Indian Kanoon headers, footers, URLs, and page markers."""
    if not text:
        return ""
    raw_lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw_lines]
    counts = Counter(line.casefold() for line in lines if line)
    kept: list[str] = []
    for line in lines:
        if not line or _is_boilerplate_line(line):
            continue
        if counts[line.casefold()] > 1 and len(line) <= 180:
            continue
        kept.append(line)
    return "\n".join(kept)


def _is_boilerplate_line(line: str) -> bool:
    lowered = line.casefold()
    patterns = (
        r"^indian kanoon\s*[-:]?\s*https?://",
        r"^https?://",
        r"^www\.",
        r"^(?:page\s*)?\d+(?:\s+of\s+\d+)?$",
        r"^page\s+\d+(?:\s+of\s+\d+)?$",
        r"^(?:downloaded|printed|generated)\s+(?:from|by)\b",
        r"^copyright\b|^all rights reserved\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def normalize_legal_abbreviations(text: str | None) -> str:
    """Expand common legal abbreviations without changing citation numbers."""
    if not text:
        return ""
    replacements = (
        (r"\bSecs?\.\s*", "Section "),
        (r"\bArts?\.\s*", "Article "),
        (r"\bvs?\.\s*", "versus "),
        (r"\bU\.\s*S\.\s*C\.\b", "USC"),
        (r"\bI\.\s*P\.\s*C\.\b", "IPC"),
        (r"\bC\.\s*P\.\s*C\.\b", "CPC"),
        (r"\bCr\.\s*P\.\s*C\.\b", "CrPC"),
    )
    normalized = text
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized


def clean_judgment_text(text: str | None) -> str:
    """Clean a judgment while preserving enough punctuation for segmentation."""
    without_boilerplate = remove_boilerplate(text)
    normalized = normalize_legal_abbreviations(without_boilerplate)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{2,}", "\n", normalized)
    return normalized.strip()


def clean_text(text: str | None) -> str:
    """Normalize whitespace while preserving the original words."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def split_sentences(text: str | None) -> list[str]:
    """Split cleaned text with pysbd, falling back when the optional package is absent."""
    cleaned = clean_judgment_text(text)
    if not cleaned:
        return []
    try:
        import pysbd
        segmenter = pysbd.Segmenter(language="en", clean=False)
        return [part.strip() for part in segmenter.segment(cleaned) if part.strip()]
    except ImportError:
        LOGGER.warning("pysbd is unavailable; using the regex sentence splitter")
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean_text(cleaned)) if part.strip()]


def word_count(text: str | None) -> int:
    """Count whitespace-delimited words in cleaned text."""
    return len(clean_text(text).split())


def document_id(record: Mapping[str, Any], text: str) -> str:
    """Return a stable source id, falling back to a content hash."""
    for key in ("doc_id", "id", "document_id", "source_path"):
        value = str(record.get(key, "")).strip()
        if value:
            return Path(value).stem if key == "source_path" else value
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def preprocess_record(record: Mapping[str, Any], text_column: str = "text") -> dict[str, Any] | None:
    """Clean one judgment and drop it when outside the configured word limits."""
    cleaned = clean_judgment_text(str(record.get(text_column, "")))
    count = word_count(cleaned)
    if count < MIN_JUDGMENT_WORDS or count > MAX_JUDGMENT_WORDS:
        return None
    return {
        "doc_id": document_id(record, cleaned),
        "cleaned_sentences": split_sentences(cleaned),
    }


def deduplicate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate cleaned judgments while preserving first-seen order."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        sentences = [str(sentence).strip() for sentence in record.get("cleaned_sentences", []) if str(sentence).strip()]
        fingerprint = hashlib.sha1(" ".join(sentences).casefold().encode("utf-8")).hexdigest()
        if sentences and fingerprint not in seen:
            seen.add(fingerprint)
            unique.append({"doc_id": str(record["doc_id"]), "cleaned_sentences": sentences})
    return unique


def preprocess_records(records: Sequence[Mapping[str, Any]], text_column: str = "text") -> list[dict[str, Any]]:
    """Clean, length-filter, sentence-split, and deduplicate judgment records."""
    processed = [result for record in records if (result := preprocess_record(record, text_column)) is not None]
    return deduplicate_records(processed)


def write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write processed records as UTF-8 JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=True) + "\n")


def create_fixed_splits(records: Sequence[Mapping[str, Any]], output_dir: Path, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Create exact 2,000/100/200 deterministic splits or fail without writing partial data."""
    required = sum(FIXED_SPLIT_SIZES.values())
    if len(records) < required:
        raise ValueError(f"Need {required} unique judgments for fixed splits; received {len(records)}")
    items = [dict(record) for record in records]
    random.Random(seed).shuffle(items)
    splits: dict[str, list[dict[str, Any]]] = {}
    start = 0
    for name, size in FIXED_SPLIT_SIZES.items():
        splits[name] = items[start : start + size]
        start += size
    for name, split in splits.items():
        write_jsonl(split, output_dir / f"{name}.jsonl")
    return splits


def build_fixed_splits(input_path: Path, output_dir: Path = Path("data/splits"), seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    """Load extracted JSONL judgments, preprocess them, and write fixed splits."""
    records = load_tabular_corpus(input_path)
    processed = preprocess_records(records)
    return create_fixed_splits(processed, output_dir, seed)


def detect_columns(rows: Sequence[Mapping[str, Any]], requested: Mapping[str, str | None] | None = None) -> dict[str, str | None]:
    """Detect id, text, and summary columns using names and content heuristics."""
    names = list(rows[0].keys()) if rows else []
    requested = requested or {}
    result: dict[str, str | None] = {}
    aliases = {
        "id": ("id", "uid", "case_id", "document_id"),
        "text": ("text", "document", "judgment", "content", "body"),
        "summary": ("summary", "abstract", "highlights", "target"),
    }
    for role, candidates in aliases.items():
        choice = requested.get(role)
        result[role] = choice if choice in names else next((name for name in names if name.lower() in candidates), None)
    return result


def load_tabular_corpus(path: Path, columns: Mapping[str, str | None] | None = None) -> list[dict[str, str]]:
    """Load CSV or JSONL records without requiring pandas."""
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, str]] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            records = [{str(k): str(v or "") for k, v in row.items()} for row in csv.DictReader(handle)]
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        import json
        with path.open("r", encoding="utf-8") as handle:
            records = [dict(json.loads(line)) for line in handle if line.strip()]
    else:
        raise ValueError("Only CSV and JSONL files are supported")
    detected = detect_columns(records, columns)
    text_column = detected.get("text")
    if text_column:
        for record in records:
            record[text_column] = clean_text(record.get(text_column, ""))
    return records


def deterministic_split(records: Sequence[Mapping[str, Any]], seed: int = 42, train_size: float = 0.8, dev_size: float = 0.1) -> dict[str, list[dict[str, Any]]]:
    """Return reproducible train/dev/test partitions."""
    if not 0 <= train_size <= 1 or not 0 <= dev_size <= 1 or train_size + dev_size > 1:
        raise ValueError("split sizes must be between zero and one and sum to at most one")
    items = [dict(record) for record in records]
    random.Random(seed).shuffle(items)
    train_end = round(len(items) * train_size)
    dev_end = train_end + round(len(items) * dev_size)
    return {"train": items[:train_end], "dev": items[train_end:dev_end], "test": items[dev_end:]}
