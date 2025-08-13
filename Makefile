.PHONY: help venv download prepare data train evaluate predict notebooks gui

# Defaults
SYS_PY := python                 # for bootstrapping venv only
PY := .venv/bin/python           # venv python for everything else
CLI := $(PY) -m src.cli
CONFIG ?= configs

help:
	@echo "Available targets:"
	@echo "  venv      Create .venv and install requirements"
	@echo "  download  Download dataset (uses configs/dataset.yaml)"
	@echo "  prepare   Create data/ and models/ directories"
	@echo "  data      Build datasets: load/split/preprocess -> data/*"
	@echo "  train     Train model and save artifacts"
	@echo "  evaluate  Evaluate saved model and generate metrics"
	@echo "  predict   Run batch predictions on a CSV (INPUT=...)"
	@echo "  notebooks Launch Jupyter Lab"
	@echo "  gui       Launch Streamlit GUI"

venv:
	$(SYS_PY) -m src.cli init-venv

download:
	$(CLI) download-data --config $(CONFIG)

prepare:
	$(CLI) prepare --config $(CONFIG)

data:
	$(CLI) build-data --config $(CONFIG)

train:
	$(CLI) train --config $(CONFIG)

evaluate:
	$(CLI) evaluate --config $(CONFIG)

predict:
	@# Usage: make predict INPUT=path/to/input.csv
	@if [ -z "$(INPUT)" ]; then \
		echo "Usage: make predict INPUT=path/to/input.csv"; \
		exit 1; \
	fi
	$(CLI) predict --config $(CONFIG) --input '$(INPUT)'

notebooks:
	jupyter lab

gui:
	streamlit run app_streamlit_full.py
