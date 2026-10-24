PY=.venv/bin/python
.PHONY: env all test
env:
	uv venv --python 3.12 .venv && uv pip install --python $(PY) -r requirements.txt
all:
	$(PY) src/select_set.py          # choose proteins -> data/selected.json (needs network)
	$(PY) scripts/download_data.py   # PDB, AlphaFold DB, ESMFold (ESM Atlas API)
	$(PY) src/evaluate.py            # per-protein / per-residue accuracy
	$(PY) src/analyze.py             # tables, summary.json, figures
test:
	$(PY) -m pytest -q tests
