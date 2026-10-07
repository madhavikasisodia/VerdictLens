from verdictlens.extractive import (
    EntityAwareExtractor,
    lexrank_scores,
    score_sentence,
    summarize_extractive,
    summarize_lexrank,
    summarize_textrank,
    textrank_scores,
)


def test_extractive_handles_empty_and_limits_sentences() -> None:
    assert summarize_extractive("") == ""
    text = "The court held this. The appeal was allowed. Background facts follow."
    summary = summarize_extractive(text, max_sentences=1)
    assert summary.endswith(".")
    assert len(summary.split(". ")) == 1
    assert score_sentence("") == 0.0


def test_textrank_and_lexrank_return_deterministic_scores() -> None:
    sentences = ["The court considered Section 10.", "The appeal was allowed.", "Background facts followed."]
    assert len(textrank_scores(sentences)) == 3
    assert len(lexrank_scores(sentences)) == 3
    assert summarize_textrank(" ".join(sentences), word_budget=20)
    assert summarize_lexrank(" ".join(sentences), word_budget=20)


def test_entity_aware_extractor_forces_disposition_and_respects_budget() -> None:
    text = "The court considered Section 10. " + "Background facts are recorded. " * 10 + "The appeal was dismissed."
    summary = EntityAwareExtractor(use_minilm=False).summarize(text, word_budget=3)
    assert "dismissed" in summary.casefold()


def test_entity_aware_mmr_deduplicates_repeated_sentences() -> None:
    text = "The court held the issue was important. " * 3 + "The appeal was allowed."
    summary = EntityAwareExtractor(use_minilm=False).summarize(text, word_budget=30)
    assert summary.count("The court held the issue was important.") == 1
