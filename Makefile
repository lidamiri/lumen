PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install test lint eval serve clean

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

test:
	$(BIN)/pytest

lint:
	$(BIN)/python -m compileall -q src tests

eval:
	$(BIN)/lumen ingest examples/docs
	$(BIN)/lumen eval examples/eval.jsonl --report reports/eval.json

serve:
	$(BIN)/lumen serve --port 8000

clean:
	rm -rf build dist *.egg-info .pytest_cache reports
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
