from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verdictlens.entities import extract_entities

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file", type=Path, required=True)
    args = parser.parse_args()
    LOGGER.info("Entities: %s", extract_entities(args.text_file.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
