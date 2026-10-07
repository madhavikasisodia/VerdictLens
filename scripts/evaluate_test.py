"""Evaluate cached generation systems on the fixed 200-judgment test set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verdictlens.evaluate import evaluate_test_systems


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all cached systems on test")
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.jsonl"))
    parser.add_argument("--generation-dir", type=Path, default=Path("outputs/generations"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--source", type=Path, default=Path("outputs/extracted_judgments.jsonl"))
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--check-support", action="store_true")
    args = parser.parse_args()
    evaluate_test_systems(args.test, args.generation_dir, args.output_dir, args.expected_count, args.check_support, args.source)


if __name__ == "__main__":
    main()
