from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verdictlens.preprocess import build_fixed_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/extracted_judgments.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        splits = build_fixed_splits(args.input, args.output_dir, args.seed)
    except ValueError as exc:
        LOGGER.error("Could not create fixed splits: %s", exc)
        raise SystemExit(1) from exc
    for name, rows in splits.items():
        LOGGER.info("Wrote %d rows to %s", len(rows), args.output_dir / f"{name}.jsonl")


if __name__ == "__main__":
    main()
