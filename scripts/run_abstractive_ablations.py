"""Run A0-A3 abstractive systems on the processed dev split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verdictlens.abstractive import SummarizerConfig, ablation_configs, generate_cached

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load generation settings from YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load generation config") from exc
    with path.open("r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def load_dev_records(path: Path) -> list[dict[str, Any]]:
    """Load processed dev records containing doc_id and cleaned_sentences."""
    if not path.exists():
        raise FileNotFoundError(f"Dev split not found: {path}. Build data/splits/dev.jsonl first.")
    with path.open("r", encoding="utf-8") as handle:
        return [dict(json.loads(line)) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run abstractive ablations A0-A3 on dev")
    parser.add_argument("--dev", type=Path, default=Path("data/splits/dev.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generations"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = load_dev_records(args.dev)
    if args.limit is not None:
        records = records[: args.limit]
    settings = load_yaml(args.config)
    configs = ablation_configs(settings)
    expected = ("A0", "A1", "A2", "A3")
    missing = [name for name in expected if name not in configs]
    if missing:
        raise ValueError(f"Generation config is missing ablations: {', '.join(missing)}")
    for ablation in expected:
        config = SummarizerConfig(**{**configs[ablation].__dict__, "output_dir": args.output_dir})
        LOGGER.info("Running %s: %s (%s)", ablation, config.model_name, config.mode)
        results = generate_cached(records, config)
        LOGGER.info("Cached %d summaries in %s", len(results), config.output_dir / f"{config.system_name}.jsonl")


if __name__ == "__main__":
    main()
