"""Entity extraction for Indian Supreme Court judgments."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LegalEntities:
    """Structured legal entities extracted from one judgment."""

    statutes: list[str] = field(default_factory=list)
    constitutional_articles: list[str] = field(default_factory=list)
    case_citations: list[str] = field(default_factory=list)
    parties: list[str] = field(default_factory=list)
    courts: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    outcome: str = ""

    @property
    def final_outcome(self) -> str:
        """Return the single final outcome label, if one was detected."""
        return self.outcome

    def all_values(self) -> set[str]:
        """Return normalized values across all entity categories."""
        values: set[str] = set()
        for field_values in (
            self.statutes,
            self.constitutional_articles,
            self.case_citations,
            self.parties,
            self.courts,
            self.dates,
            self.outcomes,
        ):
            values.update(value.casefold() for value in field_values)
        if self.outcome:
            values.add(self.outcome.casefold())
        return values


def _unique(values: list[str]) -> list[str]:
    """Normalize whitespace and preserve first-seen order."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip(" \t\r\n,;:")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
    return result


@lru_cache(maxsize=1)
def _load_spacy_model() -> Any | None:
    """Load the optional English spaCy model once per process."""
    try:
        import spacy
    except ImportError:
        LOGGER.debug("spaCy is unavailable; using regex party extraction")
        return None
    try:
        return spacy.load("en_core_web_sm")
    except (OSError, ImportError):
        LOGGER.info("spaCy model en_core_web_sm is unavailable; using regex party extraction")
        return None


def _extract_parties(text: str) -> list[str]:
    """Extract party names from spaCy PERSON/ORG entities and legal labels."""
    parties: list[str] = []
    nlp = _load_spacy_model()
    if nlp is not None:
        try:
            parties.extend(entity.text for entity in nlp(text).ents if entity.label_ in {"PERSON", "ORG"})
        except Exception as exc:  # spaCy pipelines can fail on malformed input.
            LOGGER.debug("spaCy party extraction failed: %s", exc)

    label_pattern = re.compile(
        r"\b(?:PETITIONER|RESPONDENT|APPELLANT|APPELLANTS|APPLICANT|ACCUSED|COMPLAINANT|DEFENDANT|PLAINTIFF)\s*:?\s*([^\n.;]+)",
        re.IGNORECASE,
    )
    parties.extend(match.group(1) for match in label_pattern.finditer(text))

    versus_pattern = re.compile(
        r"\b([A-Z][A-Za-z0-9.&'() -]{1,80}?)\s+(?:v\.?s?\.?|versus)\s+([A-Z][A-Za-z0-9.&'() -]{1,100}?)(?=\s*[,.(\n]|$)",
        re.IGNORECASE,
    )
    for match in versus_pattern.finditer(text):
        parties.extend(match.groups())
    return _unique(parties)


def extract_outcome_label(text: str | None, final_window: bool = True) -> str:
    """Find the rightmost outcome phrase, limited to the final 15% by default."""
    text = text or ""
    if not text:
        return ""
    window = text[int(len(text) * 0.85) :] if final_window else text
    labels = (
        ("partly allowed", "partly allowed"),
        ("set aside", "set aside"),
        ("upheld", "upheld"),
        ("remanded", "remanded"),
        ("convicted", "convicted"),
        ("acquitted", "acquitted"),
        ("dismissed", "dismissed"),
        ("allowed", "allowed"),
    )
    matches: list[tuple[int, int, str]] = []
    for phrase, label in labels:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", window, re.IGNORECASE):
            matches.append((match.start(), match.end(), label))
    matches = [
        candidate
        for candidate in matches
        if not any(
            other[0] <= candidate[0]
            and other[1] >= candidate[1]
            and (other[1] - other[0]) > (candidate[1] - candidate[0])
            for other in matches
        )
    ]
    return max(matches, default=(-1, -1, ""))[2]


def extract_entities(text: str | None) -> LegalEntities:
    """Extract statutes, citations, parties, courts, dates, and final outcome."""
    source = text or ""
    statutes = _unique(
        re.findall(
            r"\bSection\s+[0-9A-Za-z()/-]+(?:\s+(?:IPC|CrPC|CPC)|\s+of\s+(?:the\s+)?[A-Za-z][A-Za-z0-9.&' -]*(?:Act|Code))?",
            source,
            re.IGNORECASE,
        )
    )
    articles = _unique(re.findall(r"\bArticle\s+[0-9A-Za-z()/-]+\b", source, re.IGNORECASE))
    citations = _unique(
        re.findall(
            r"(?:\(\s*(?:19|20)\d{2}\s*\)\s*\d+\s+(?:SCC|SCR)\s+\d+|\bAIR\s+(?:19|20)\d{2}\s+[A-Z][A-Za-z .&-]*?\s+\d+|\b(?:19|20)\d{2}\s+(?:SCC|SCR)\s+\d+)",
            source,
            re.IGNORECASE,
        )
    )
    court_pattern = r"\b(?:Supreme Court(?: of India)?|High Court(?: of [A-Z][A-Za-z .&-]+)?|District Court|Constitution Bench|Court of Appeal|Court of [A-Z][A-Za-z .&-]+)\b"
    courts = _unique(re.findall(court_pattern, source, re.IGNORECASE))
    dates = _unique(
        re.findall(
            r"\b(?:[0-3]?\d[/-][01]?\d[/-](?:19|20)?\d{2}|[0-3]?\d\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+(?:19|20)\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+[0-3]?\d,?\s+(?:19|20)\d{2}|(?:19|20)\d{2}-[01]?\d-[0-3]?\d)\b",
            source,
            re.IGNORECASE,
        )
    )
    outcome = extract_outcome_label(source)
    return LegalEntities(
        statutes=statutes,
        constitutional_articles=articles,
        case_citations=citations,
        parties=_extract_parties(source),
        courts=courts,
        dates=dates,
        outcomes=[outcome] if outcome else [],
        outcome=outcome,
    )
