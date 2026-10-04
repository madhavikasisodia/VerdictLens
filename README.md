# VerdictLens

**Intelligent Summarization of Indian Court Judgments**

VerdictLens is a research-oriented B.Tech project for exploring modular NLP and machine learning workflows over Indian court judgments. The repository is intentionally scaffolded without ML implementations or runtime dependencies so that each component can be developed, tested, and evaluated independently.

## Repository Structure

```text
VerdictLens/
|
|-- backend/
|   |-- app/
|   |   |-- extraction.py       # PDF and plain-text extraction boundary
|   |   |-- preprocessing.py    # Legal document cleaning and segmentation
|   |   |-- summarization.py    # Extractive and abstractive summarization boundary
|   |   |-- entities.py         # Legal entity extraction boundary
|   |   |-- consistency.py      # Factual consistency verification boundary
|   |   |-- evaluation.py       # ROUGE, BERTScore, and other metrics boundary
|   |   `-- __init__.py
|   |-- models/                 # Model loading and model-specific adapters
|   |-- services/               # Application services and orchestration
|   |-- utils/                  # Shared, reusable utilities
|   |-- main.py                 # Future backend entry point
|   `-- requirements.txt        # Backend dependency placeholder
|
|-- frontend/                  # Future Next.js + TypeScript application
|-- data/
|   |-- raw/                    # Original judgments; keep out of version control
|   |-- processed/              # Derived datasets and intermediate artifacts
|   `-- samples/                # Small, shareable development samples
|-- experiments/                # Reproducible experiment configurations and notes
|-- evaluation/                 # Evaluation runs, reports, and reference summaries
|-- notebooks/                  # Exploratory research notebooks
|-- tests/                      # Unit and integration tests
|-- requirements.txt            # Root Python dependency placeholder
|-- README.md
`-- .gitignore
```

## Planned Architecture

The backend will expose small, independently testable boundaries:

1. **Extraction** converts PDF or text inputs into a normalized document representation.
2. **Preprocessing** cleans legal text and identifies useful sections or segments.
3. **Summarization** provides separate extractive and Transformer-based abstractive strategies behind a common interface.
4. **Entity extraction** identifies legal entities such as courts, judges, parties, statutes, dates, and case references.
5. **Consistency verification** checks generated summaries against source judgments for factual support.
6. **Evaluation** calculates ROUGE, BERTScore, and additional research metrics without coupling them to model code.
7. **Services** coordinates these components for future API or command-line workflows.
8. **Models** contains model configuration, loading, and adapters as implementations are added.

The frontend will eventually provide a Next.js and TypeScript interface for uploading judgments, requesting summaries, reviewing extracted entities, and viewing evaluation or verification results. It is currently only a placeholder and has no frontend dependencies.

## Python Virtual Environment

From the repository root on Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install dependencies only after they are intentionally added to `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

To leave the environment:

```text
deactivate
```

The virtual environment directory is ignored by Git. A future backend setup may use a separate environment or dependency lock strategy if the research workflow requires it.

## Current Status

This repository contains structure and documentation only. PDF extraction, NLP preprocessing, summarization, entity recognition, factual verification, evaluation, API endpoints, and frontend functionality are intentionally not implemented yet.

## Development Principles

- Keep each NLP capability modular and independently testable.
- Keep raw and generated data separate from source code.
- Record experiments and evaluation results reproducibly.
- Add dependencies only when a concrete implementation requires them.
- Do not commit confidential or copyrighted court-document datasets without checking their usage rights.
