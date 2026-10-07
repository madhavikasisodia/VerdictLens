"""Fine-tune LED on judgment-summary JSONL data."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "nsi319/legal-led-base-16384"
MAX_SOURCE_LENGTH = 4096
MAX_TARGET_LENGTH = 512


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML training configuration."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load training configuration") from exc
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def load_records(path: Path) -> list[dict[str, str]]:
    """Load JSONL records and require non-empty reference summaries."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = dict(json.loads(line))
            sentences = raw.get("cleaned_sentences", [])
            text = " ".join(str(sentence) for sentence in sentences) if isinstance(sentences, list) and sentences else str(raw.get("text", ""))
            summary = next((str(raw.get(key, "")).strip() for key in ("summary", "reference_summary", "target", "abstract") if str(raw.get(key, "")).strip()), "")
            if text.strip() and summary:
                records.append({"doc_id": str(raw.get("doc_id", raw.get("id", len(records)))), "text": text, "summary": summary})
    if not records:
        raise ValueError(f"No records with non-empty reference summaries found in {path}")
    return records


def extract_headnote(text: str) -> str:
    """Extract an embedded HEADNOTE section as a weak reference summary."""
    match = re.search(r"\bHEADNOTE\s*:\s*(.*?)(?=\n\s*JUDGMENT\b|\Z)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    lines = []
    for line in match.group(1).splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned and not re.match(r"^(?:Indian Kanoon|https?://|www\.)", cleaned, re.IGNORECASE):
            lines.append(cleaned)
    return " ".join(lines).strip()


def prepare_headnote_splits(source_path: Path, train_path: Path, dev_path: Path, seed: int = 42) -> tuple[int, int]:
    """Create deterministic weakly supervised train/dev files from extracted PDFs."""
    if not source_path.exists():
        raise FileNotFoundError(f"Cannot prepare splits; source corpus not found: {source_path}")
    records: list[dict[str, str]] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = dict(json.loads(line))
            text = str(raw.get("text", ""))
            summary = extract_headnote(text)
            if text.strip() and summary:
                records.append({"doc_id": str(raw.get("id", len(records))), "text": text, "summary": summary})
    if len(records) < 2:
        raise ValueError("Could not find at least two embedded HEADNOTE summaries for train/dev preparation")
    random.Random(seed).shuffle(records)
    dev_count = max(1, len(records) // 10)
    dev_records = records[:dev_count]
    train_records = records[dev_count:]
    for path, rows in ((train_path, train_records), (dev_path, dev_records)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    LOGGER.warning("Prepared weakly supervised splits from embedded HEADNOTE sections: train=%d dev=%d", len(train_records), len(dev_records))
    return len(train_records), len(dev_records)


def tokenize_records(records: Iterable[dict[str, str]], tokenizer: Any, max_source_length: int = MAX_SOURCE_LENGTH, max_target_length: int = MAX_TARGET_LENGTH) -> list[dict[str, Any]]:
    """Tokenize source and target text with fixed source and target ceilings."""
    tokenized: list[dict[str, Any]] = []
    for record in records:
        encoded = tokenizer(record["text"], max_length=max_source_length, truncation=True, padding=False)
        labels = tokenizer(text_target=record["summary"], max_length=max_target_length, truncation=True, padding=False)
        encoded["labels"] = labels["input_ids"]
        tokenized.append(encoded)
    return tokenized


class LEDDataCollator:
    """Pad batches and set LED global attention on the first token."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        try:
            from transformers import DataCollatorForSeq2Seq
        except ImportError as exc:
            raise RuntimeError("transformers is required for fine-tuning") from exc
        self._collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self._collator(features)
        attention_mask = batch["attention_mask"]
        global_attention_mask = attention_mask.new_zeros(attention_mask.shape)
        global_attention_mask[:, 0] = 1
        batch["global_attention_mask"] = global_attention_mask
        return batch


class VRAMLoggingCallback:
    """Log peak CUDA memory at major trainer lifecycle events."""

    def _log(self, event: str) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
                LOGGER.info("VRAM %s: allocated=%.2f GB peak=%.2f GB", event, allocated, peak)
        except ImportError:
            return

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._log("train_begin")

    def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._log("epoch_end")

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._log("evaluate")

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._log("train_end")


def build_training_arguments(output_dir: Path, config: dict[str, Any], fp16: bool) -> Any:
    """Build version-compatible Transformers TrainingArguments."""
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise RuntimeError("transformers is required for fine-tuning") from exc
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "num_train_epochs": float(config.get("num_train_epochs", 3)),
        "learning_rate": float(config.get("learning_rate", 3e-5)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.05)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "fp16": fp16,
        "gradient_checkpointing": True,
        "save_strategy": "epoch",
        "logging_steps": int(config.get("logging_steps", 10)),
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "seed": int(config.get("seed", 42)),
    }
    evaluation_key = "eval_strategy" if "eval_strategy" in parameters else "evaluation_strategy"
    kwargs[evaluation_key] = "epoch"
    return TrainingArguments(**kwargs)


def train(args: argparse.Namespace) -> dict[str, float]:
    """Fine-tune LED and evaluate loss on the dev set."""
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, set_seed
    except ImportError as exc:
        raise RuntimeError("Install torch and transformers before fine-tuning") from exc
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    if args.auto_prepare and (not args.train.exists() or not args.dev.exists()):
        prepare_headnote_splits(args.source, args.train, args.dev, int(config.get("seed", 42)))
    train_records = load_records(args.train)
    dev_records = load_records(args.dev)
    model_name = str(config.get("model_name", DEFAULT_MODEL))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if bool(config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    use_fp16 = bool(config.get("fp16", True)) and torch.cuda.is_available()
    if bool(config.get("fp16", True)) and not use_fp16:
        LOGGER.warning("CUDA unavailable; running without fp16")
    train_dataset = tokenize_records(train_records, tokenizer, int(config.get("max_source_length", MAX_SOURCE_LENGTH)), int(config.get("max_target_length", MAX_TARGET_LENGTH)))
    dev_dataset = tokenize_records(dev_records, tokenizer, int(config.get("max_source_length", MAX_SOURCE_LENGTH)), int(config.get("max_target_length", MAX_TARGET_LENGTH)))
    training_args = build_training_arguments(args.output_dir, config, use_fp16)
    collator = LEDDataCollator(tokenizer, model)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": collator,
        "callbacks": [VRAMLoggingCallback()],
    }
    trainer_parameters = inspect.signature(Seq2SeqTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_parameters else "tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    resume = args.resume_from_checkpoint
    if resume is None:
        try:
            from transformers.trainer_utils import get_last_checkpoint
            resume = get_last_checkpoint(str(args.output_dir))
        except ImportError:
            resume = None
    LOGGER.info("Starting training; resume_from_checkpoint=%s", resume or "none")
    trainer.train(resume_from_checkpoint=resume)
    metrics = {key: float(value) for key, value in trainer.evaluate().items() if isinstance(value, (int, float))}
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Dev metrics: %s", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune LED on legal judgment summaries")
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/splits/dev.jsonl"))
    parser.add_argument("--source", type=Path, default=Path("outputs/extracted_judgments.jsonl"), help="Raw extracted JSONL used for automatic headnote-based preparation")
    parser.add_argument("--auto-prepare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/led_finetune"))
    parser.add_argument("--metrics-output", type=Path, default=Path("outputs/results/led_finetune_metrics.json"))
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
