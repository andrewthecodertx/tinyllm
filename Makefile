PY := .venv/bin/python
DATA := data

.PHONY: help venv info train generate test clean

help:
	@echo "make venv      create .venv and install dependencies"
	@echo "make install   install the CPU-only PyTorch build"
	@echo "make info      report corpus and model size"
	@echo "make train     train on $(DATA)"
	@echo "make generate  sample from the checkpoint"
	@echo "make test      run the test suite"
	@echo "make clean     remove caches and checkpoints"
	@echo "make reset     rebuild .venv from scratch"

venv:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install pytest

install:
	$(PY) -m pip install --index-url https://download.pytorch.org/whl/cpu torch

info:
	$(PY) tiny-llm.py info --data-dir $(DATA)

train:
	$(PY) -u tiny-llm.py train --data-dir $(DATA)

generate:
	$(PY) tiny-llm.py generate --prompt "Astronomy Notes" --tokens 400

test:
	$(PY) -m pytest tests/ -q

clean:
	rm -rf __pycache__ tests/__pycache__ .pytest_cache checkpoints/*.pt

reset:
	rm -rf .venv
	$(MAKE) venv