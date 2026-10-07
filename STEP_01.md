# Step 01: Environment and Dataset Setup

1. Create and activate a Python 3.11 virtual environment.
2. Install `requirements.txt`.
3. Run `python scripts/check_env.py` to inspect Python, optional packages, and the hardware limits.
4. Place explicitly downloaded judgment data under `data/raw/`.
5. Run `python scripts/download_and_inspect.py --path data/raw/<file>` to inspect a local CSV or JSONL file. The script does not download data implicitly.

The configured deterministic seed is 42. Keep raw data unchanged; write derived files to `data/splits/` or `outputs/`.
