"""Extractive summarization baselines and entity-aware selection."""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .entities import LegalEntities, extract_entities
from .preprocess import split_sentences

LOGGER = logging.getLogger(__name__)
LEGAL_CUES = {
    "held", "hold", "holding", "therefore", "accordingly", "hence", "issue", "issues",
    "court", "judgment", "appeal", "petition", "concluded", "find", "found", "reason",
    "because", "liable", "jurisdiction", "constitutional", "statute", "section", "article",
}
DISPOSITION_RE = re.compile(
    r"\b(?:partly\s+allowed|set\s+aside|upheld|remanded|convicted|acquitted|dismissed|allowed)\b",
    re.IGNORECASE,
)


def _tokens(sentence: str) -> Counter[str]:
    return Counter(re.findall(r"[A-Za-z][A-Za-z0-9'-]*", sentence.casefold()))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _similarity_matrix(sentences: Sequence[str]) -> list[list[float]]:
    vectors = [_tokens(sentence) for sentence in sentences]
    return [[_cosine(left, right) if index != other else 0.0 for other, right in enumerate(vectors)] for index, left in enumerate(vectors)]


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0 if maximum else 0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def textrank_scores(sentences: Sequence[str], iterations: int = 30, damping: float = 0.85) -> list[float]:
    """Return TextRank scores from a cosine sentence-similarity graph."""
    if not sentences:
        return []
    graph = _similarity_matrix(sentences)
    scores = [1.0] * len(sentences)
    for _ in range(iterations):
        updated: list[float] = []
        for index, row in enumerate(graph):
            incoming = 0.0
            for source, weight in enumerate(row):
                degree = sum(graph[source])
                if degree:
                    incoming += weight * scores[source] / degree
            updated.append((1.0 - damping) + damping * incoming)
        scores = updated
    return _normalize_scores(scores)


def lexrank_scores(sentences: Sequence[str], threshold: float = 0.1) -> list[float]:
    """Return LexRank scores using thresholded cosine sentence similarity."""
    if not sentences:
        return []
    graph = _similarity_matrix(sentences)
    scores = []
    for index, row in enumerate(graph):
        neighbors = [weight for weight in row if weight >= threshold]
        scores.append(sum(neighbors) / len(neighbors) if neighbors else 0.0)
    return _normalize_scores(scores)


def _is_disposition(sentence: str) -> bool:
    return bool(DISPOSITION_RE.search(sentence))


def _word_count(sentence: str) -> int:
    return len(sentence.split())


def _select_ranked_sentences(
    sentences: Sequence[str],
    scores: Sequence[float],
    word_budget: int = 3000,
    similarity: Sequence[Sequence[float]] | None = None,
    mmr_lambda: float = 0.7,
) -> list[str]:
    """Select sentences under a word budget while forcing disposition coverage."""
    if word_budget < 1 or not sentences:
        return []
    chosen: list[int] = []
    used_words = 0
    disposition_indices = [index for index, sentence in enumerate(sentences) if _is_disposition(sentence)]
    for index in disposition_indices:
        words = _word_count(sentences[index])
        if index not in chosen:
            chosen.append(index)
            used_words += words
    remaining = set(range(len(sentences))) - set(chosen)
    while remaining:
        candidates: list[tuple[float, int]] = []
        for index in remaining:
            words = _word_count(sentences[index])
            if used_words + words > word_budget:
                continue
            diversity = max((similarity[index][selected] for selected in chosen), default=0.0) if similarity else 0.0
            if diversity >= 0.95:
                continue
            mmr = mmr_lambda * scores[index] - (1.0 - mmr_lambda) * diversity
            candidates.append((mmr, index))
        if not candidates:
            break
        _, selected = max(candidates, key=lambda item: (item[0], -item[1]))
        chosen.append(selected)
        used_words += _word_count(sentences[selected])
        remaining.remove(selected)
    return [sentences[index] for index in sorted(chosen)]


def summarize_textrank(text: str | None, word_budget: int = 3000) -> str:
    """Summarize with the TextRank baseline under a word budget."""
    sentences = split_sentences(text)
    return " ".join(_select_ranked_sentences(sentences, textrank_scores(sentences), word_budget))


def summarize_lexrank(text: str | None, word_budget: int = 3000) -> str:
    """Summarize with the LexRank baseline under a word budget."""
    sentences = split_sentences(text)
    return " ".join(_select_ranked_sentences(sentences, lexrank_scores(sentences), word_budget))


@dataclass
class EntityAwareExtractor:
    """Entity-aware extractor with optional local MiniLM centrality scoring."""

    model_name: str = "all-MiniLM-L6-v2"
    use_minilm: bool = True
    mmr_lambda: float = 0.7
    device: str | None = None

    def _centrality_scores(self, sentences: Sequence[str]) -> list[float]:
        if not sentences:
            return []
        if self.use_minilm:
            try:
                from sentence_transformers import SentenceTransformer
                kwargs = {"device": self.device} if self.device else {}
                model = SentenceTransformer(self.model_name, local_files_only=True, **kwargs)
                embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
                centrality = [float(sum(embeddings[index] @ embeddings[other] for other in range(len(sentences)) if other != index) / max(len(sentences) - 1, 1)) for index in range(len(sentences))]
                del model
                return _normalize_scores(centrality)
            except (ImportError, OSError, TypeError, ValueError) as exc:
                LOGGER.info("MiniLM unavailable locally; using lexical centrality: %s", exc)
        matrix = _similarity_matrix(sentences)
        return _normalize_scores([sum(row) / max(len(row) - 1, 1) for row in matrix])

    def score_sentences(self, sentences: Sequence[str], entities: LegalEntities | None = None) -> list[float]:
        """Score sentences by centrality, legal cues, entity density, and position."""
        if not sentences:
            return []
        entities = entities or extract_entities(" ".join(sentences))
        entity_values = [value.casefold() for values in (
            entities.statutes,
            entities.constitutional_articles,
            entities.case_citations,
            entities.parties,
            entities.courts,
            entities.dates,
        ) for value in values]
        centrality = self._centrality_scores(sentences)
        raw: list[float] = []
        for index, sentence in enumerate(sentences):
            lowered = sentence.casefold()
            words = max(_word_count(sentence), 1)
            cue_density = sum(1 for token in _tokens(sentence) if token in LEGAL_CUES) / words
            entity_density = sum(lowered.count(value) for value in entity_values) / words
            position = 1.0 - (index / max(len(sentences) - 1, 1))
            disposition_bonus = 1.0 if _is_disposition(sentence) else 0.0
            raw.append(0.45 * centrality[index] + 0.2 * cue_density + 0.2 * min(entity_density, 1.0) + 0.05 * position + 0.1 * disposition_bonus)
        return _normalize_scores(raw)

    def select_sentences(self, sentences: Sequence[str], word_budget: int = 3000) -> list[str]:
        """Select scored sentences with MMR de-duplication."""
        entities = extract_entities(" ".join(sentences))
        scores = self.score_sentences(sentences, entities)
        return _select_ranked_sentences(sentences, scores, word_budget, _similarity_matrix(sentences), self.mmr_lambda)

    def summarize(self, text: str | None, word_budget: int = 3000) -> str:
        """Return an entity-aware extractive summary."""
        sentences = split_sentences(text)
        return " ".join(self.select_sentences(sentences, word_budget))


def score_sentence(sentence: str, entities: LegalEntities | None = None) -> float:
    """Score one sentence using the entity-aware heuristic path."""
    return EntityAwareExtractor(use_minilm=False).score_sentences([sentence], entities)[0] if sentence.strip() else 0.0


def summarize_extractive(text: str | None, max_sentences: int = 5) -> str:
    """Backward-compatible top-sentence summarizer."""
    if max_sentences < 1:
        return ""
    sentences = split_sentences(text)
    entities = extract_entities(text)
    extractor = EntityAwareExtractor(use_minilm=False)
    scores = extractor.score_sentences(sentences, entities)
    chosen = sorted(range(len(sentences)), key=lambda index: (-scores[index], index))[:max_sentences]
    return " ".join(sentences[index] for index in sorted(chosen))
