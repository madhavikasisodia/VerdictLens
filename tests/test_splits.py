from __future__ import annotations

import pandas as pd
import pytest

from src.data import splits


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5", "doc-6"],
            "text": [
                "Alpha judgment",
                "Beta judgment",
                "Gamma judgment",
                "Delta judgment",
                "Epsilon judgment",
                "Zeta judgment",
            ],
            "summary": ["A", "B", "C", "D", "E", "F"],
        }
    )


def test_make_splits_has_no_document_overlap(sample_frame: pd.DataFrame, tmp_path) -> None:
    splits.SPLITS_PATH = tmp_path / "splits.json"

    result = splits.make_splits(
        sample_frame,
        seed=42,
        ratios={"train": 0.8, "val": 0.1, "test": 0.1},
    )

    split_sets = [set(doc_ids) for doc_ids in result.values()]
    assert set.union(*split_sets) == set(sample_frame["doc_id"])
    for index, current in enumerate(split_sets):
        for other in split_sets[index + 1 :]:
            assert current.isdisjoint(other)


def test_make_splits_is_deterministic(sample_frame: pd.DataFrame, tmp_path) -> None:
    splits.SPLITS_PATH = tmp_path / "splits.json"
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

    first = splits.make_splits(sample_frame, seed=42, ratios=ratios)
    second = splits.make_splits(sample_frame, seed=42, ratios=ratios, force=True)

    assert first == second


def test_make_splits_refuses_to_overwrite(sample_frame: pd.DataFrame, tmp_path) -> None:
    splits.SPLITS_PATH = tmp_path / "splits.json"
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    splits.make_splits(sample_frame, seed=42, ratios=ratios)

    with pytest.raises(FileExistsError):
        splits.make_splits(sample_frame, seed=42, ratios=ratios)
