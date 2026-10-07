# VerdictLens

VerdictLens is an entity-grounded, agentic verify-and-repair summarization system for Indian Supreme Court judgments. It combines legal entity extraction, extractive and abstractive summarization, sentence-level verification, evidence-constrained repair, and experiment reporting.

## Current Data

The local corpus contains 519 judgment PDFs from 1950-1954. Raw PDFs, extracted text, split files, model checkpoints, traces, and generated reports are intentionally ignored by Git. They must be supplied or regenerated locally.

The PDFs contain embedded headnotes but no independent human reference-summary column. The training and test preparation helpers therefore use extracted headnotes as weak references. Treat results from these weak labels as development evidence, not gold-standard evaluation.

## Setup

Use Python 3.11 and the workspace virtual environment:

```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional evaluation and local-repair services:

- `bert-score` enables BERTScore metrics.
- Ollama must be running locally with `qwen2.5:7b-instruct` available for agent repair.
- Hugging Face model weights are downloaded only when an explicit generation or training command is run.

## Tests and Environment

```powershell
.venv\\Scripts\\Activate.ps1
python scripts/check_env.py
python -m pytest -q
```

## Data Workflow

Inspect the supplied PDFs and extract text:

```powershell
python scripts/download_and_inspect.py --path data/raw/supreme_court_judgments
```

This writes `outputs/extracted_judgments.jsonl` and `outputs/dataset_report.md`.

Preprocess judgments and create fixed 2,000/100/200 splits when enough unique documents exist:

```powershell
python scripts/make_splits.py
```

The current corpus has fewer than 2,300 qualifying judgments, so the fixed split command refuses to fabricate records. Fine-tuning and evaluation can instead prepare weak headnote-based train/dev/test files automatically.

## Summarization

Run extractive coverage experiments:

```powershell
python scripts/evaluate_extractive.py --input data/splits/dev.jsonl
```

Run abstractive A0-A3 systems. This requires local access to the configured Hugging Face models:

```powershell
python scripts/run_abstractive_ablations.py --dev data/splits/dev.jsonl
```

Generation caches are written to `outputs/generations/` as one JSONL file per system.

Fine-tune LED with epoch checkpoints, resume support, fp16 on CUDA, gradient checkpointing, and dev loss evaluation:

```powershell
python scripts/finetune_led.py
```

Training uses embedded headnotes as weak summaries when `data/splits/train.jsonl` and `data/splits/dev.jsonl` are absent. Checkpoints go under `outputs/led_finetune/` and metrics under `outputs/results/`.

Evaluate cached systems on the 200-record test set:

```powershell
python scripts/evaluate_test.py
```

The evaluator writes CSV and LaTeX tables to `outputs/results/`. It reports P7 faithfulness metrics, ROUGE-1/2/L, optional BERTScore, compression ratio, runtime, bootstrap 95% confidence intervals, and paired A4-vs-A3/A3-vs-A2 tests.

## Agent CLI

Run one judgment through the LangGraph pipeline:

```powershell
python src/verdictlens/agent.py --text-file path\\to\\judgment.txt
```

The graph runs extract, generate, verify, repair, verify, and finalize. Repair is capped at two loops, uses Ollama evidence-constrained sentence edits, and records node inputs and outputs under `outputs/traces/`.

## Layout

```text
configs/                 YAML data, generation, and training settings
data/raw/                Local source PDFs; ignored by Git
data/splits/             Derived JSONL splits; ignored by Git
outputs/                 Extracted data, caches, checkpoints, traces, reports
scripts/                 Dataset, training, ablation, and evaluation CLIs
src/verdictlens/         Reusable preprocessing and modeling package
tests/                   Focused pytest coverage
```

## Reproducibility and Limits

The configured seed is 42. Model workflows target batch size 1, a maximum source length of 4096 tokens, fp16 where CUDA supports it, gradient checkpointing where available, and one GPU model at a time. Generated artifacts and large local inputs are excluded by [.gitignore](.gitignore).

This project is a research prototype. It does not provide legal advice, and generated summaries require human review.
