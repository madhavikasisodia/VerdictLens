"""Inspect configured datasets without modifying their contents."""

from __future__ import annotations

import json
import math
import numbers
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "experiments" / "eda" / "dataset_report.md"
MAX_STRING_LENGTH = 300
WORD_LIMIT = 4096

sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (  # noqa: E402
    load_iltur_dataset,
    load_iltur_rr_dataset,
    load_in_abs_dataset,
)


def _truncate(value: Any) -> Any:
    """Recursively truncate strings for readable example output."""
    if isinstance(value, str):
        if len(value) <= MAX_STRING_LENGTH:
            return value
        return f"{value[:MAX_STRING_LENGTH]}..."
    if isinstance(value, dict):
        return {str(key): _truncate(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_truncate(item) for item in value]
    return value


def _json_value(value: Any) -> str:
    """Render a value without failing on dataset-specific scalar types."""
    try:
        return json.dumps(_truncate(value), ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(_truncate(value)), ensure_ascii=False)


def _is_dataset_dict(value: Any) -> bool:
    return value.__class__.__name__ == "DatasetDict" and hasattr(value, "items")


def _split_views(value: Any) -> list[tuple[str, Any]]:
    """Return named views for split containers and single tabular objects."""
    if _is_dataset_dict(value) or isinstance(value, dict):
        return [(str(name), split) for name, split in value.items()]
    return [("data", value)]


def _records_and_columns(value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract records and discovered columns from supported object shapes."""
    if isinstance(value, pd.DataFrame):
        columns = [str(column) for column in value.columns]
        return value.to_dict(orient="records"), columns

    if hasattr(value, "column_names") and hasattr(value, "__getitem__"):
        columns = [str(column) for column in value.column_names]
        records = value[:]
        if isinstance(records, dict):
            row_count = len(next(iter(records.values()), []))
            return [
                {column: records[column][index] for column in columns}
                for index in range(row_count)
            ], columns

    if isinstance(value, list):
        records = [record for record in value if isinstance(record, dict)]
        columns = sorted({str(key) for record in records for key in record})
        return records, columns

    if isinstance(value, dict):
        return [value], [str(key) for key in value]

    raise TypeError(f"Unsupported dataset object: {type(value).__name__}")


def _column_values(records: list[dict[str, Any]], column: str) -> list[Any]:
    return [record.get(column) for record in records]


def _flatten_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        strings: list[str] = []
        for item in value:
            strings.extend(_flatten_strings(item))
        return strings
    return []


def _text_for_stats(value: Any) -> str:
    return " ".join(_flatten_strings(value))


def _flatten_scalars(value: Any) -> list[Any]:
    if isinstance(value, dict):
        scalars: list[Any] = []
        for item in value.values():
            scalars.extend(_flatten_scalars(item))
        return scalars
    if isinstance(value, (list, tuple)):
        scalars: list[Any] = []
        for item in value:
            scalars.extend(_flatten_scalars(item))
        return scalars
    return [value]


def _dtype(value: Any, column: str, records: list[dict[str, Any]]) -> str:
    features = getattr(value, "features", None)
    if features is not None and column in features:
        return str(features[column])
    return str(pd.Series(_column_values(records, column), name=column).dtype)


def _is_text_field(values: list[Any]) -> bool:
    texts = [_text_for_stats(value) for value in values if _text_for_stats(value)]
    if not texts:
        return False
    word_counts = [len(text.split()) for text in texts]
    return max(word_counts) >= 5 or max(map(len, texts)) >= 40


def _is_label_field(values: list[Any]) -> bool:
    comparable = [
        scalar
        for value in values
        for scalar in _flatten_scalars(value)
        if scalar is not None and not isinstance(scalar, float)
    ]
    if not comparable:
        return False
    unique_count = len({_json_value(value) for value in comparable})
    return unique_count <= min(50, max(2, math.ceil(len(comparable) * 0.05)))


def _format_text_stats(values: list[Any]) -> str:
    word_counts = [len(_text_for_stats(value).split()) for value in values if _text_for_stats(value)]
    if not word_counts:
        return "No non-empty text values detected."
    over_limit = sum(count > WORD_LIMIT for count in word_counts)
    return (
        f"min={min(word_counts)}, median={median(word_counts):g}, max={max(word_counts)} words; "
        f"examples over {WORD_LIMIT} words={over_limit}"
    )


def _format_label_counts(values: list[Any]) -> str:
    counts = Counter(
        _json_value(scalar)
        for value in values
        for scalar in _flatten_scalars(value)
        if scalar is not None
    )
    return ", ".join(f"{label}: {count}" for label, count in counts.most_common())


def _inspect_view(name: str, value: Any) -> list[str]:
    records, columns = _records_and_columns(value)
    lines = [f"### Split: `{name}`", "", f"- Return type: `{type(value).__name__}`", f"- Size: `{len(records)}`", ""]
    lines.append("#### Columns and dtypes")
    lines.append("")
    lines.append("| Column | Dtype |")
    lines.append("| --- | --- |")
    for column in columns:
        lines.append(f"| `{column}` | `{_dtype(value, column, records)}` |")

    examples = records[:2]
    lines.extend(["", "#### First 2 examples", ""])
    if examples:
        lines.append("```json")
        lines.append(_json_value(examples))
        lines.append("```")
    else:
        lines.append("No examples available.")

    text_columns = [column for column in columns if _is_text_field(_column_values(records, column))]
    lines.extend(["", "#### Text-field statistics", ""])
    if text_columns:
        for column in text_columns:
            lines.append(f"- `{column}`: {_format_text_stats(_column_values(records, column))}")
    else:
        lines.append("No text-like fields detected.")

    label_columns = [column for column in columns if _is_label_field(_column_values(records, column))]
    lines.extend(["", "#### Label value counts", ""])
    if label_columns:
        for column in label_columns:
            lines.append(f"- `{column}`: {_format_label_counts(_column_values(records, column))}")
    else:
        lines.append("No label-like fields detected.")
    return lines


def inspect_dataset(name: str, loader: Any) -> list[str]:
    """Load and inspect one dataset, returning Markdown report lines."""
    lines = [f"## {name}", ""]
    try:
        value = loader()
        lines.append(f"Return type: `{type(value).__name__}`")
        lines.append("")
        if _is_dataset_dict(value) or isinstance(value, dict):
            lines.append(f"Split names: `{', '.join(str(key) for key in value)}`")
            lines.append("")
        for split_name, split in _split_views(value):
            lines.extend(_inspect_view(split_name, split))
            lines.append("")
    except Exception as error:  # Keep the report useful when one dataset is unavailable.
        lines.extend([f"Inspection failed: `{type(error).__name__}: {error}`", ""])
    return lines


def main() -> None:
    report_lines = [
        "# Dataset Inspection Report",
        "",
        "Generated by `scripts/inspect_datasets.py`. Dataset objects are read only.",
        "",
    ]
    loaders = [
        (
            "IN-Abs",
            partial(load_in_abs_dataset, PROJECT_ROOT / "configs" / "base.yaml"),
        ),
        (
            "IL-TUR CJPE",
            partial(load_iltur_dataset, PROJECT_ROOT / "configs" / "base.yaml"),
        ),
        (
            "IL-TUR RR",
            partial(load_iltur_rr_dataset, PROJECT_ROOT / "configs" / "base.yaml"),
        ),
    ]
    for name, loader in loaders:
        report_lines.extend(inspect_dataset(name, loader))

    report = "\n".join(report_lines).rstrip() + "\n"
    print(report, end="")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
