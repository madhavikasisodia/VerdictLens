"""Debug Hub refs, files, and loading modes for the IL-TUR dataset."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "experiments" / "iltur_debug.md"
REPO_ID = "Exploration-Lab/IL-TUR"


class Report:
    """Print report lines and retain them for the Markdown output file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def save(self) -> None:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(self.lines).rstrip() + "\n", encoding="utf-8")


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _ref_name(ref: Any) -> str:
    return str(getattr(ref, "name", ref))


def _check_load(
    report: Report,
    label: str,
    loader: Callable[[], Any],
) -> None:
    try:
        loader()
    except Exception as error:  # Each loading mode must be reported independently.
        report.write(f"{label}: FAIL - {type(error).__name__}: {error}")
    else:
        report.write(f"{label}: PASS")


def main() -> None:
    report = Report()
    report.write(f"datasets version: {_version('datasets')}")
    report.write("# IL-TUR Debug Report")
    report.write("")
    report.write(f"Repository: `{REPO_ID}`")
    report.write("")

    api = HfApi()
    report.write("## Hub refs")
    try:
        refs = api.list_repo_refs(REPO_ID, repo_type="dataset")
        report.write("### Branches")
        for branch in refs.branches:
            report.write(f"- {_ref_name(branch)}")
        report.write("### Tags")
        for tag in refs.tags:
            report.write(f"- {_ref_name(tag)}")
        report.write("### Converts")
        for convert in refs.converts:
            report.write(f"- {_ref_name(convert)}")
    except Exception as error:
        report.write(f"Hub refs: FAIL - {type(error).__name__}: {error}")

    report.write("")
    report.write("## Repository files")
    try:
        files = api.list_repo_files(REPO_ID, repo_type="dataset")
        for file_name in files:
            report.write(f"- {file_name}")
    except Exception as error:
        report.write(f"Repository files: FAIL - {type(error).__name__}: {error}")

    for config_name in ("cjpe", "rr"):
        report.write("")
        report.write(f"## Config: `{config_name}`")
        _check_load(
            report,
            "revision=script",
            lambda config_name=config_name: load_dataset(
                REPO_ID,
                config_name,
                revision="script",
            ),
        )
        _check_load(
            report,
            "default revision",
            lambda config_name=config_name: load_dataset(REPO_ID, config_name),
        )
        _check_load(
            report,
            "revision=script, trust_remote_code=True",
            lambda config_name=config_name: load_dataset(
                REPO_ID,
                config_name,
                revision="script",
                trust_remote_code=True,
            ),
        )

    report.save()


if __name__ == "__main__":
    main()
