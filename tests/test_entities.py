import pytest

from verdictlens.entities import LegalEntities, extract_entities


EXAMPLES = [
    (
        "The Supreme Court considered Section 302 of the Indian Penal Code and Article 21. The appeal was allowed.",
        "allowed",
        "Section 302",
        "Article 21",
    ),
    (
        "In (2019) 3 SCC 100 and AIR 2018 SC 55, the High Court of Delhi dismissed the petition.",
        "dismissed",
        "(2019) 3 SCC 100",
        "High Court of Delhi",
    ),
    (
        "Section 34 IPC was applied. The Supreme Court set aside the conviction and remanded the matter.",
        "remanded",
        "Section 34 IPC",
        "Supreme Court",
    ),
    (
        "The petitioner challenged Article 14. The final order partly allowed the appeal.",
        "partly allowed",
        "Article 14",
        "partly allowed",
    ),
    (
        "The case appears at 1950 SCR 88. The order of the High Court was upheld.",
        "upheld",
        "1950 SCR 88",
        "High Court",
    ),
    (
        "The matter under Section 138 of the Negotiable Instruments Act ended with acquittal. The accused was acquitted.",
        "acquitted",
        "Section 138 of the Negotiable Instruments Act",
        "acquitted",
    ),
    (
        "A.K. Gopalan v. The State of Madras. PETITIONER: A.K. GOPALAN RESPONDENT: THE STATE OF MADRAS. The appeal was dismissed.",
        "dismissed",
        "A.K. Gopalan",
        "The State of Madras",
    ),
    (
        "The District Court referred the dispute to the Constitution Bench. The accused was convicted.",
        "convicted",
        "District Court",
        "Constitution Bench",
    ),
    (
        "The judgment dated 19 May 1950 cites AIR 1950 SC 27. The decree was set aside.",
        "set aside",
        "AIR 1950 SC 27",
        "19 May 1950",
    ),
    (
        "The appeal was allowed at the beginning, but after reconsideration the Supreme Court dismissed it.",
        "dismissed",
        "Supreme Court",
        "dismissed",
    ),
]


@pytest.mark.parametrize("text,outcome,expected,other", EXAMPLES)
def test_realistic_legal_examples(text: str, outcome: str, expected: str, other: str) -> None:
    result = extract_entities(text + " The final order was " + outcome + ".")
    combined = (
        result.statutes
        + result.constitutional_articles
        + result.case_citations
        + result.parties
        + result.courts
        + result.dates
        + result.outcomes
    )
    assert any(expected.casefold() in value.casefold() for value in combined)
    assert any(other.casefold() in value.casefold() for value in combined)
    assert result.outcome == outcome
    assert result.final_outcome == outcome


def test_empty_entities_are_stable() -> None:
    result = extract_entities("")
    assert result == LegalEntities()
    assert result.all_values() == set()
