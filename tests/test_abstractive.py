from pathlib import Path

import pytest

from verdictlens.abstractive import (
    SummarizerConfig,
    ablation_configs,
    cache_path,
    cleanup_model,
    generate_cached,
    generate_summary,
    read_generation_cache,
    _generation_kwargs,
    _token_chunks,
)


def test_config_limits_and_empty_input() -> None:
    config = SummarizerConfig()
    assert config.max_source_length == 4096
    assert config.batch_size == 1
    assert generate_summary("", config) == ""
    cleanup_model(None)


def test_missing_transformers_fails_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def blocked(name: str, *args: object, **kwargs: object):
        if name == "transformers":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    with pytest.raises(RuntimeError, match="transformers"):
        generate_summary("A legal judgment.")


def test_token_chunks_respect_size_and_overlap() -> None:
    chunks = _token_chunks(list(range(10)), chunk_size=4, overlap=1)
    assert chunks == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9], [9]]
    with pytest.raises(ValueError):
        _token_chunks([1], 2, 2)


def test_generation_kwargs_use_beam_settings() -> None:
    kwargs = _generation_kwargs(SummarizerConfig(num_beams=6, min_new_tokens=200, max_new_tokens=100))
    assert kwargs["num_beams"] == 6
    assert kwargs["min_new_tokens"] == 100
    assert kwargs["no_repeat_ngram_size"] == 3


def test_ablation_configs_define_all_four_systems() -> None:
    configs = ablation_configs({
        "systems": {
            "A0": {"name": "led_truncated", "model_name": "led", "mode": "truncated"},
            "A1": {"name": "led_chunked", "model_name": "led", "mode": "chunked"},
            "A2": {"name": "pegasus_truncated", "model_name": "pegasus", "mode": "truncated"},
            "A3": {"name": "pegasus_chunked", "model_name": "pegasus", "mode": "chunked"},
        }
    })
    assert set(configs) == {"A0", "A1", "A2", "A3"}
    assert configs["A3"].mode == "chunked"


def test_generation_cache_reuses_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = SummarizerConfig(system_name="test_system", output_dir=tmp_path)
    cache_path(config).write_text('{"doc_id": "doc-1", "summary": "cached"}\n', encoding="utf-8")
    monkeypatch.setattr("verdictlens.abstractive.generate_summary", lambda text, config: "new")
    records = [{"doc_id": "doc-1", "cleaned_sentences": ["A judgment."]}, {"doc_id": "doc-2", "cleaned_sentences": ["Another judgment."]}]
    results = generate_cached(records, config)
    assert results[0]["summary"] == "cached"
    assert results[1]["summary"] == "new"
    assert read_generation_cache(config)["doc-2"] == "new"
