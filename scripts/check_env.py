from __future__ import annotations

import importlib.util
import logging
import platform
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    LOGGER.info("Python: %s", sys.version.split()[0])
    LOGGER.info("Platform: %s", platform.platform())
    for name in ("yaml", "pytest", "torch", "transformers", "rouge_score"):
        LOGGER.info("%s available: %s", name, importlib.util.find_spec(name) is not None)
    LOGGER.info("Limits: max 8 GB VRAM, max 24 GB RAM, batch size 1, source length 4096")


if __name__ == "__main__":
    main()
