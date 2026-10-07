# Step 02: Preprocessing and Entity Extraction

Use `verdictlens.preprocess.load_tabular_corpus` for CSV or JSONL input, `detect_columns` for id/text/summary inference, and `deterministic_split` for reproducible train/dev/test partitions.

Use `verdictlens.entities.extract_entities` to identify statutes, constitutional articles, case citations, parties, courts, dates, and outcomes. The baseline extractor is deterministic and regex-based, so review its output before using it as training metadata.

Examples:

```powershell
python scripts/make_splits.py --input data/raw/judgments.csv
python scripts/entity_demo.py --text-file data/raw/example.txt
```
