"""Run train-only exploratory analysis for the IN-Abs dataset."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import load_split  # noqa: E402

LOGGER = logging.getLogger(__name__)
EDA_DIR = PROJECT_ROOT / "experiments" / "eda"
THRESHOLDS = [1024, 2048, 4096, 8192, 16384]
POSITION_SAMPLE_SIZE = 500


def _word_count(value: str) -> int:
    return len(re.findall(r"\b\w+\b", value))


def _token_count(tokenizer: AutoTokenizer, value: str, special_tokens: bool = True) -> int:
    return len(
        tokenizer(
            value,
            add_special_tokens=special_tokens,
            truncation=False,
        )["input_ids"]
    )


def _token_lengths(
    tokenizer: AutoTokenizer,
    values: Iterable[str],
    batch_size: int = 32,
) -> list[int]:
    values_list = list(values)
    lengths = []
    for start in range(0, len(values_list), batch_size):
        batch = values_list[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=True, truncation=False)
        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])
    return lengths


def _split_sentences(value: str) -> list[str]:
    return [sentence.strip() for sentence in nltk.sent_tokenize(value) if sentence.strip()]


def _ensure_punkt_resources() -> None:
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab/english",
    }
    for package, resource in resources.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, quiet=True):
                raise LookupError(f"Unable to download required NLTK resource: {package}")


def _save_length_analysis(
    train: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> pd.DataFrame:
    lengths = train[["doc_id", "text", "summary"]].copy()
    lengths["text_words"] = lengths["text"].map(_word_count)
    lengths["summary_words"] = lengths["summary"].map(_word_count)
    lengths["text_tokens"] = _token_lengths(tokenizer, lengths["text"])
    lengths["summary_tokens"] = _token_lengths(tokenizer, lengths["summary"])
    length_table = lengths.drop(columns=["text", "summary"])
    length_table.to_csv(EDA_DIR / "length_distributions.csv", index=False)

    threshold_table = pd.DataFrame(
        {
            "threshold_tokens": THRESHOLDS,
            "percentage_judgments_over": [
                float((lengths["text_tokens"] > threshold).mean() * 100)
                for threshold in THRESHOLDS
            ],
        }
    )
    threshold_table.to_csv(EDA_DIR / "token_threshold_percentages.csv", index=False)
    print(threshold_table.to_string(index=False))

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    plots = [
        ("text_words", "Text word lengths", "Words"),
        ("summary_words", "Summary word lengths", "Words"),
        ("text_tokens", "Text LED token lengths", "Tokens"),
        ("summary_tokens", "Summary LED token lengths", "Tokens"),
    ]
    for axis, (column, title, label) in zip(axes.flat, plots):
        axis.hist(lengths[column], bins=40, color="#174c63", alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel(label)
        axis.set_ylabel("Documents")
    figure.tight_layout()
    figure.savefig(EDA_DIR / "length_distributions.png", dpi=150)
    plt.close(figure)
    return lengths


def _save_compression_analysis(lengths: pd.DataFrame) -> pd.DataFrame:
    compression = lengths.copy()
    compression["compression_ratio"] = (
        compression["summary_words"] / compression["text_words"]
    )
    compression["compression_ratio"].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).rename("compression_ratio").to_csv(EDA_DIR / "compression_percentiles.csv", header=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(compression["compression_ratio"], bins=50, color="#2a7f62", alpha=0.85)
    axis.set_title("Summary-to-text compression ratio")
    axis.set_xlabel("Summary words / text words")
    axis.set_ylabel("Documents")
    figure.tight_layout()
    figure.savefig(EDA_DIR / "compression_ratio_histogram.png", dpi=150)
    plt.close(figure)

    category_masks = {
        "ratio_gt_0.8": compression["compression_ratio"] > 0.8,
        "ratio_lt_0.02": compression["compression_ratio"] < 0.02,
        "summary_words_gt_3000": compression["summary_words"] > 3000,
    }
    outlier_rows = []
    for category, mask in category_masks.items():
        category_rows = compression.loc[mask, [
            "doc_id",
            "text_words",
            "summary_words",
            "compression_ratio",
        ]].copy()
        category_rows.insert(0, "category", category)
        outlier_rows.append(category_rows)
        print(f"\n{category} samples:")
        print(category_rows.head(3).to_string(index=False))
    outliers = pd.concat(outlier_rows, ignore_index=True)
    outliers.to_csv(EDA_DIR / "outliers.csv", index=False)
    return compression


def _position_rows(
    sample: pd.DataFrame,
    tokenizer: AutoTokenizer,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for document in sample.itertuples(index=False):
        source_sentences = _split_sentences(document.text)
        summary_sentences = _split_sentences(document.summary)
        if not source_sentences or not summary_sentences:
            continue

        source_token_lengths = [
            _token_count(tokenizer, sentence, special_tokens=False)
            for sentence in source_sentences
        ]
        source_starts = np.cumsum([0] + source_token_lengths[:-1]).tolist()
        source_vectors = TfidfVectorizer().fit_transform(source_sentences)
        summary_vectors = TfidfVectorizer().fit(source_sentences).transform(summary_sentences)
        similarities = cosine_similarity(summary_vectors, source_vectors)
        length_bucket = (
            "<2k"
            if document.text_tokens < 2000
            else "2k-4k"
            if document.text_tokens <= 4000
            else ">4k"
        )
        for summary_index, similarity_row in enumerate(similarities):
            best_source_index = int(np.argmax(similarity_row))
            rows.append(
                {
                    "doc_id": document.doc_id,
                    "summary_sentence_index": summary_index,
                    "source_sentence_index": best_source_index,
                    "relative_position": (
                        best_source_index / max(len(source_sentences) - 1, 1)
                    ),
                    "source_start_token": source_starts[best_source_index],
                    "source_tokens": document.text_tokens,
                    "length_bucket": length_bucket,
                    "cosine_similarity": float(similarity_row[best_source_index]),
                }
            )
    return pd.DataFrame(rows)


def _save_position_analysis(train: pd.DataFrame, tokenizer: AutoTokenizer, seed: int) -> None:
    sample = train.sample(n=min(POSITION_SAMPLE_SIZE, len(train)), random_state=seed)
    rows = _position_rows(sample, tokenizer)
    if rows.empty:
        raise ValueError("No sentence alignments were produced from the train sample")
    rows.to_csv(EDA_DIR / "position_bias_alignments.csv", index=False)

    summary_rows = []
    for bucket, group in [("overall", rows), *rows.groupby("length_bucket")]:
        summary_rows.append(
            {
                "length_bucket": bucket,
                "alignments": len(group),
                "fraction_source_start_over_1024": float(
                    (group["source_start_token"] > 1024).mean()
                ),
                "fraction_source_start_over_4096": float(
                    (group["source_start_token"] > 4096).mean()
                ),
            }
        )
    position_summary = pd.DataFrame(summary_rows)
    position_summary.to_csv(EDA_DIR / "position_bias_summary.csv", index=False)
    print("\nPosition-bias summary:")
    print(position_summary.to_string(index=False))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    panels = [("overall", rows), *rows.groupby("length_bucket")]
    for axis, (bucket, group) in zip(axes.flat, panels):
        axis.hist(group["relative_position"], bins=20, range=(0, 1), color="#8b3a62")
        axis.set_title(f"{bucket} ({len(group):,} summary sentences)")
        axis.set_xlabel("Relative source position")
        axis.set_ylabel("Summary sentences")
        axis.set_xlim(0, 1)
    figure.tight_layout()
    figure.savefig(EDA_DIR / "position_bias_histograms.png", dpi=150)
    plt.close(figure)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    config_path = PROJECT_ROOT / "configs" / "base.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    seed = int(config.get("seed", 42))
    tokenizer_name = config.get("models", {}).get("led", "allenai/led-base-16384")

    train = load_split("train")
    if train.empty:
        raise ValueError("The train split is empty")
    print(f"Loaded {len(train):,} train documents")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    _ensure_punkt_resources()
    lengths = _save_length_analysis(train, tokenizer)
    compression = _save_compression_analysis(lengths)
    train_with_lengths = train.merge(
        lengths[["doc_id", "text_tokens"]], on="doc_id", how="inner", validate="one_to_one"
    )
    _save_position_analysis(train_with_lengths, tokenizer, seed)
    print(f"EDA outputs saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
