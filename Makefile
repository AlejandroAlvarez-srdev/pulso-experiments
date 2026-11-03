PY ?= .venv/bin/python
NOTEBOOKS := $(wildcard notebooks/*.ipynb)

.PHONY: install lint test notebooks html all

install:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e .

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

test:
	$(PY) -m pytest

# Re-execute every notebook in place so the committed outputs are fresh.
notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS)

html:
	$(PY) -m jupyter nbconvert --to html --output-dir docs $(NOTEBOOKS)

all: lint test notebooks html
