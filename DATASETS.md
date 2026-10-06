# Datasets

VerdictLens uses three dataset loading targets from the Hugging Face Hub. The datasets are loaded at runtime with the `datasets` library; no dataset files are committed to this repository.

## 1. IN-Abs

**Source:** `percins/IN-ABS`

IN-Abs contains Indian court judgments paired with reference summaries. It is used for summarization experiments and exploratory analysis.

The loader maps the available fields into the project's canonical columns:

- `doc_id`: source file identifier
- `text`: judgment text
- `summary`: reference summary
- `year`: unavailable and left empty
- `court`: unavailable and left empty

Load it with:

```python
from src.data.loader import load_in_abs_dataset

in_abs = load_in_abs_dataset()
```

## 2. IL-TUR CJPE

**Source:** `Exploration-Lab/IL-TUR`

**Configuration:** `cjpe`

**Revision:** `script`

CJPE is the judgment prediction and explanation task. Its records include judgment text, an acceptance/rejection label, and expert rankings or explanations.

Available splits:

- `single_train`
- `single_dev`
- `multi_train`
- `multi_dev`
- `test`
- `expert`

Load it with:

```python
from src.data.loader import load_iltur_dataset

cjpe = load_iltur_dataset()
single_train = cjpe["single_train"]
```

## 3. IL-TUR RR

**Source:** `Exploration-Lab/IL-TUR`

**Configuration:** `rr`

**Revision:** `script`

RR is the rhetorical-role classification task. It labels sections of Indian legal judgments with roles such as facts, issues, arguments, statutes, precedents, and rulings.

Available splits:

- `CL_train`
- `CL_dev`
- `CL_test`
- `IT_train`
- `IT_dev`
- `IT_test`

Load it with:

```python
from src.data.loader import load_iltur_rr_dataset

rr = load_iltur_rr_dataset()
cl_train = rr["CL_train"]
it_train = rr["IT_train"]
```

## Access and Usage Notes

- `IN-ABS` is publicly available on the Hugging Face Hub.
- `IL-TUR` is gated and may require accepting its research-use terms and authenticating with Hugging Face.
- The first call to `load_dataset()` may download and cache data locally. Later calls can reuse the local cache.
- IL-TUR is released for research use and is not intended for commercial use. Check the dataset card for the current license and terms before redistribution or deployment.

## Loader Configuration

The dataset settings are maintained in [`configs/base.yaml`](configs/base.yaml), and the loading functions are implemented in [`src/data/loader.py`](src/data/loader.py).
