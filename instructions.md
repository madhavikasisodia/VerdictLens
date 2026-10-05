Project: VerdictLens, a research project on summarizing Indian court judgments
(two-stage: role-aware extractive -> LED/Pegasus abstractive -> verify-and-repair).
Rules: Python 3.10+, type hints, docstrings, PyTorch + HuggingFace Transformers.
All hyperparameters live in YAML files in configs/, never hardcoded.
Set seeds everywhere. Use pathlib. Log with the logging module. Every module needs
a pytest test in tests/. Legal documents are long (5k-50k tokens): never assume
they fit a model window; always chunk explicitly. Do not invent dataset schemas;
read them from the files.