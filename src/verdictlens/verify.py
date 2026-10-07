"""Faithfulness verification for legal summaries."""

from __future__ import annotations

import gc
import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any, Sequence

from .entities import extract_entities, extract_outcome_label
from .preprocess import split_sentences

LOGGER = logging.getLogger(__name__)
FOREIGN_PATTERNS = (
    r"\bU\.?\s*S\.?\s*C\.?\b",
    r"\bUnited States Code\b",
    r"\bCircuit Court(?:s)?\b",
    r"\bCourt of Appeals?\b",
    r"\bSupreme Court of the United States\b",
    r"\bU\.?\s*K\.?\b",
    r"\bUnited Kingdom\b",
    r"\bHouse of Lords\b",
)


@dataclass(frozen=True)
class SentenceSupportFlag:
    """A summary sentence that failed or could not receive support."""

    sentence: str
    reasons: list[str] = field(default_factory=list)
    retrieved_source_sentences: list[str] = field(default_factory=list)
    entailment_score: float | None = None
    entailment_label: str | None = None


@dataclass(frozen=True)
class VerificationReport:
    """Structured faithfulness findings for one generated summary."""

    unsupported_entities: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    outcome_inconsistencies: list[str] = field(default_factory=list)
    entity_precision: float = 1.0
    statute_precision: float = 1.0
    statute_recall: float = 1.0
    foreign_leakage: list[str] = field(default_factory=list)
    outcome_match: bool | None = None
    p3_outcome: str | None = None
    sentence_flags: list[SentenceSupportFlag] = field(default_factory=list)
    support_model_status: str = "not_run"

    @property
    def is_faithful(self) -> bool:
        """Whether no unsupported, contradictory, leakage, or support issue was found."""
        return not (
            self.unsupported_entities
            or self.outcome_inconsistencies
            or self.foreign_leakage
            or self.sentence_flags
        )

    @property
    def flagged_sentences(self) -> list[SentenceSupportFlag]:
        """Alias for callers that use the report terminology directly."""
        return self.sentence_flags


def _precision_recall(summary_values: Sequence[str], source_values: Sequence[str]) -> tuple[float, float]:
    summary_set = {value.casefold() for value in summary_values}
    source_set = {value.casefold() for value in source_values}
    overlap = summary_set & source_set
    precision = len(overlap) / len(summary_set) if summary_set else 1.0
    recall = len(overlap) / len(source_set) if source_set else 1.0
    return precision, recall


def _foreign_leakage(summary: str) -> list[str]:
    """Return distinct non-Indian legal terms found in the summary."""
    findings: list[str] = []
    for pattern in FOREIGN_PATTERNS:
        findings.extend(match.group(0) for match in re.finditer(pattern, summary, re.IGNORECASE))
    return list(dict.fromkeys(value.casefold() for value in findings))


def _cleanup_embedding_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _retrieve_top_source_sentences(source_sentences: Sequence[str], summary_sentence: str, top_k: int) -> list[str]:
    """Retrieve source sentences using local MiniLM cosine similarity."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is unavailable") from exc
    model = None
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        embeddings = model.encode([summary_sentence, *source_sentences], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        scores = embeddings[1:] @ embeddings[0]
        indices = sorted(range(len(source_sentences)), key=lambda index: float(scores[index]), reverse=True)[:top_k]
        return [source_sentences[index] for index in indices]
    finally:
        if model is not None:
            _cleanup_embedding_model(model)


def _support_flags(source: str, summary: str, top_k: int = 5) -> tuple[list[SentenceSupportFlag], str]:
    """Retrieve source candidates and classify support with DeBERTa NLI."""
    source_sentences = split_sentences(source)
    summary_sentences = split_sentences(summary)
    if not source_sentences or not summary_sentences:
        return [], "no_sentences"
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return [], "sentence-transformers-unavailable"
    retrieved: dict[str, list[str]] = {}
    try:
        for sentence in summary_sentences:
            retrieved[sentence] = _retrieve_top_source_sentences(source_sentences, sentence, top_k)
        cross_encoder = CrossEncoder("cross-encoder/nli-deberta-v3-base", local_files_only=True)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.info("Sentence support models unavailable locally: %s", exc)
        return [], "support-model-unavailable"
    try:
        flags: list[SentenceSupportFlag] = []
        for sentence in summary_sentences:
            candidates = retrieved[sentence]
            predictions = cross_encoder.predict([(candidate, sentence) for candidate in candidates], apply_softmax=True)
            entailment_index = _entailment_index(cross_encoder)
            entailment_scores = [float(row[entailment_index]) for row in predictions]
            best_score = max(entailment_scores, default=0.0)
            if best_score < 0.5:
                flags.append(SentenceSupportFlag(sentence, ["no_entailing_source_sentence"], candidates, best_score, "unsupported"))
        return flags, "ok"
    finally:
        _cleanup_embedding_model(cross_encoder)


def _entailment_index(cross_encoder: Any) -> int:
    """Find the entailment class index from model metadata."""
    labels = getattr(getattr(cross_encoder, "model", None), "config", None)
    id2label = getattr(labels, "id2label", {}) or {}
    for index, label in id2label.items():
        if "entail" in str(label).casefold():
            return int(index)
    return 2


def verify_summary(source: str, summary: str, p3_outcome: str | None = None, check_support: bool = True, top_k: int = 5) -> VerificationReport:
    """Verify entities, leakage, P3 outcome agreement, and sentence-level support."""
    source_entities = extract_entities(source)
    summary_entities = extract_entities(summary)
    source_outcome = source_entities.outcome or extract_outcome_label(source, final_window=False)
    summary_outcome = extract_outcome_label(summary, final_window=False)
    if source_outcome:
        source_entities = replace(source_entities, outcomes=[source_outcome], outcome=source_outcome)
    if summary_outcome:
        summary_entities = replace(summary_entities, outcomes=[summary_outcome], outcome=summary_outcome)

    source_values = source_entities.all_values() - {value.casefold() for value in source_entities.outcomes}
    summary_values = summary_entities.all_values() - {value.casefold() for value in summary_entities.outcomes}
    unsupported = sorted(value for value in summary_values if value not in source_values)
    missing = sorted(value for value in source_values if value not in summary_values)
    entity_precision = len(summary_values & source_values) / len(summary_values) if summary_values else 1.0
    statute_values = source_entities.statutes + source_entities.constitutional_articles
    summary_statutes = summary_entities.statutes + summary_entities.constitutional_articles
    statute_precision, statute_recall = _precision_recall(summary_statutes, statute_values)
    source_outcomes = {value.casefold() for value in source_entities.outcomes}
    summary_outcomes = {value.casefold() for value in summary_entities.outcomes}
    inconsistencies = sorted(summary_outcomes - source_outcomes)
    expected_outcome = p3_outcome.casefold() if p3_outcome else None
    outcome_match = (summary_outcome.casefold() == expected_outcome) if expected_outcome and summary_outcome else (not expected_outcome)
    flags, support_status = _support_flags(source, summary, top_k) if check_support else ([], "disabled")
    return VerificationReport(
        unsupported_entities=unsupported,
        missing_entities=missing,
        outcome_inconsistencies=inconsistencies,
        entity_precision=entity_precision,
        statute_precision=statute_precision,
        statute_recall=statute_recall,
        foreign_leakage=_foreign_leakage(summary),
        outcome_match=outcome_match,
        p3_outcome=p3_outcome,
        sentence_flags=flags,
        support_model_status=support_status,
    )
