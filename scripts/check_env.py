"""Check the runtime environment and configured dataset loaders."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _first_example(value: Any) -> Any:
    """Return the first example from a tabular or split dataset object."""
    if hasattr(value, "iloc"):
        if len(value) == 0:
            raise ValueError("dataset is empty")
        return value.iloc[0]

    if hasattr(value, "items"):
        for split_name, split in value.items():
            if len(split) > 0:
                return split[0]
        raise ValueError("all dataset splits are empty")

    if len(value) == 0:
        raise ValueError("dataset is empty")
    return value[0]


def _check_dataset(name: str, loader: Callable[[], Any]) -> bool:
    try:
        _first_example(loader())
    except Exception as error:  # A failed environment check should not hide later checks.
        print(f"{name}: FAIL - {type(error).__name__}: {error}")
        return False
    print(f"{name}: PASS")
    return True


def main() -> int:
    import torch
    from huggingface_hub import get_token

    print(f"Python version: {platform.python_version()}")
    print(f"OS: {platform.platform()}")
    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA availability: {cuda_available}")

    if cuda_available:
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        free_vram, total_vram = torch.cuda.mem_get_info(0)
        print(f"Total VRAM (GB): {total_vram / (1024**3):.2f}")
        print(f"Free VRAM (GB): {free_vram / (1024**3):.2f}")
    else:
        print("GPU name: unavailable")
        print("Total VRAM (GB): unavailable")
        print("Free VRAM (GB): unavailable")

    print(f"transformers version: {_package_version('transformers')}")
    print(f"datasets version: {_package_version('datasets')}")
    print(f"Hugging Face token available: {get_token() is not None}")

    from src.data.loader import (
        load_iltur_dataset,
        load_iltur_rr_dataset,
        load_in_abs_dataset,
    )
    config_path = PROJECT_ROOT / "configs" / "base.yaml"

    checks_passed = all(
        (
            _check_dataset(
                "load_in_abs_dataset",
                lambda: load_in_abs_dataset(config_path),
            ),
            _check_dataset(
                "load_iltur_dataset",
                lambda: load_iltur_dataset(config_path),
            ),
            _check_dataset(
                "load_iltur_rr_dataset",
                lambda: load_iltur_rr_dataset(config_path),
            ),
        )
    )
    return 0 if checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
