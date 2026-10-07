from pathlib import Path

import pytest

from verdictlens.preprocess import (
    clean_judgment_text,
    clean_text,
    create_fixed_splits,
    deduplicate_records,
    detect_columns,
    deterministic_split,
    load_tabular_corpus,
    preprocess_records,
    split_sentences,
)


def test_clean_and_split_empty_and_normal() -> None:
    assert clean_text("  A\n B  ") == "A B"
    assert split_sentences("") == []
    assert split_sentences("One. Two!") == ["One.", "Two!"]


def test_detect_and_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("case_id,judgment,summary\n1,  Text here  ,Short\n", encoding="utf-8")
    rows = load_tabular_corpus(path)
    assert detect_columns(rows)["text"] == "judgment"
    assert rows[0]["judgment"] == "Text here"


def test_deterministic_split_and_invalid_sizes() -> None:
    rows = [{"id": str(i)} for i in range(10)]
    assert deterministic_split(rows) == deterministic_split(rows)
    with pytest.raises(ValueError):
        deterministic_split(rows, train_size=0.8, dev_size=0.3)


def test_cleaning_removes_boilerplate_and_expands_abbreviations() -> None:
    text = "Title\nIndian Kanoon - http://indiankanoon.org/doc/1 1\nPage 2\nSec. 302 and Art. 21 apply."
    cleaned = clean_judgment_text(text)
    assert "Indian Kanoon" not in cleaned
    assert "Page 2" not in cleaned
    assert "Section 302" in cleaned
    assert "Article 21" in cleaned


def test_preprocess_filters_deduplicates_and_uses_doc_id() -> None:
    body = "The court considered the appeal. " * 500
    records = [{"id": "one", "text": body}, {"id": "duplicate", "text": body}, {"id": "short", "text": "Too short."}]
    processed = preprocess_records(records)
    assert len(processed) == 1
    assert processed[0]["doc_id"] == "one"
    assert processed[0]["cleaned_sentences"]


def test_fixed_splits_are_exact_and_deterministic(tmp_path: Path) -> None:
    records = [{"doc_id": str(index), "cleaned_sentences": ["A sentence."]} for index in range(2300)]
    first = create_fixed_splits(records, tmp_path / "first")
    second = create_fixed_splits(records, tmp_path / "second")
    assert {name: len(rows) for name, rows in first.items()} == {"train": 2000, "dev": 100, "test": 200}
    assert first == second
    assert (tmp_path / "first" / "train.jsonl").exists()


def test_fixed_splits_refuse_insufficient_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2300"):
        create_fixed_splits([], tmp_path)
