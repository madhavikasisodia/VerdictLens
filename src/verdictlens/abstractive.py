"""Resource-aware abstractive summarization with truncation and chunking."""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

LOGGER = logging.getLogger(__name__)
MAX_SOURCE_LENGTH = 4096
BATCH_SIZE = 1
LED_MODEL = "nsi319/legal-led-base-16384"
PEGASUS_MODEL = "nsi319/legal-pegasus"


@dataclass
class SummarizerConfig:
    """Generation and resource settings for one summarization system."""

    system_name: str = "led_truncated"
    model_name: str = LED_MODEL
    mode: str = "truncated"
    max_source_length: int = MAX_SOURCE_LENGTH
    max_new_tokens: int = 180
    min_new_tokens: int = 40
    num_beams: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    fp16: bool = True
    batch_size: int = BATCH_SIZE
    chunk_overlap_tokens: int = 0
    output_dir: Path = Path("outputs/generations")


def cleanup_model(model: Any = None) -> None:
    """Release a model and clear CUDA cache between model stages."""
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        LOGGER.debug("torch is unavailable; skipped CUDA cleanup")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _load_components(config: SummarizerConfig) -> tuple[Any, Any, Any]:
    """Load tokenizer and model lazily for one generation stage."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Optional dependency 'transformers' is required for abstractive generation") from exc
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Optional dependency 'torch' is required for abstractive generation") from exc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    model.to(device)
    if config.fp16 and device.type == "cuda":
        model.half()
    model.eval()
    return tokenizer, model, device


def _token_chunks(token_ids: list[int], chunk_size: int, overlap: int = 0) -> list[list[int]]:
    """Split token ids into bounded overlapping chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap_tokens must be non-negative and smaller than chunk size")
    if not token_ids:
        return []
    step = chunk_size - overlap
    return [token_ids[start : start + chunk_size] for start in range(0, len(token_ids), step)]


def _generation_kwargs(config: SummarizerConfig) -> dict[str, Any]:
    """Return configured deterministic beam-search arguments."""
    return {
        "max_new_tokens": config.max_new_tokens,
        "min_new_tokens": min(config.min_new_tokens, config.max_new_tokens),
        "num_beams": config.num_beams,
        "length_penalty": config.length_penalty,
        "no_repeat_ngram_size": config.no_repeat_ngram_size,
        "early_stopping": True,
    }


def _generate_once(tokenizer: Any, model: Any, device: Any, token_ids: list[int], config: SummarizerConfig) -> str:
    """Generate one summary from already bounded token ids."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Optional dependency 'torch' is required for abstractive generation") from exc
    inputs = tokenizer.prepare_for_model(token_ids, return_tensors="pt", truncation=True, max_length=config.max_source_length)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **_generation_kwargs(config))
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def generate_summary(text: str, config: SummarizerConfig | None = None) -> str:
    """Generate a truncated or chunk-merged summary with one model loaded at a time."""
    if not text.strip():
        return ""
    config = config or SummarizerConfig()
    if config.mode not in {"truncated", "chunked"}:
        raise ValueError("mode must be 'truncated' or 'chunked'")
    tokenizer = model = device = None
    try:
        tokenizer, model, device = _load_components(config)
        token_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if config.mode == "truncated":
            return _generate_once(tokenizer, model, device, token_ids[: config.max_source_length], config)
        chunks = _token_chunks(token_ids, config.max_source_length, config.chunk_overlap_tokens)
        chunk_summaries = [_generate_once(tokenizer, model, device, chunk, config) for chunk in chunks]
        merged = "\n".join(summary for summary in chunk_summaries if summary)
        if not merged:
            return ""
        merged_ids = tokenizer.encode(merged, add_special_tokens=True, truncation=False)
        return _generate_once(tokenizer, model, device, merged_ids[: config.max_source_length], config)
    finally:
        cleanup_model(model)


def cache_path(config: SummarizerConfig) -> Path:
    """Return the JSONL cache path for a system name."""
    return config.output_dir / f"{config.system_name}.jsonl"


def read_generation_cache(config: SummarizerConfig) -> dict[str, str]:
    """Read cached summaries keyed by document id."""
    path = cache_path(config)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return {str(row["doc_id"]): str(row.get("summary", "")) for row in (json.loads(line) for line in handle if line.strip())}


def generate_cached(records: Iterable[Mapping[str, Any]], config: SummarizerConfig) -> list[dict[str, str]]:
    """Generate summaries, reusing and updating the per-system JSONL cache."""
    cached = read_generation_cache(config)
    results: list[dict[str, str]] = []
    for record in records:
        doc_id = str(record["doc_id"])
        text = " ".join(str(sentence) for sentence in record.get("cleaned_sentences", []))
        summary = cached.get(doc_id)
        if summary is None:
            summary = generate_summary(text, config)
            cached[doc_id] = summary
        results.append({"doc_id": doc_id, "summary": summary, "system_name": config.system_name, "model_name": config.model_name})
    path = cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    return results


def ablation_configs(settings: Mapping[str, Any] | None = None) -> dict[str, SummarizerConfig]:
    """Build A0-A3 configurations from generation settings."""
    settings = dict(settings or {})
    systems = settings.get("systems", {})
    common = {key: settings[key] for key in ("max_source_length", "max_new_tokens", "min_new_tokens", "num_beams", "length_penalty", "no_repeat_ngram_size", "fp16", "batch_size", "chunk_overlap_tokens") if key in settings}
    return {
        ablation: SummarizerConfig(system_name=spec["name"], model_name=spec["model_name"], mode=spec["mode"], **common)
        for ablation, spec in systems.items()
    }
