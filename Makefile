.PHONY: help venv download prepare data train evaluate notebooks

# Defaults
PY ?= python
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
	@echo "  notebooks Launch Jupyter Lab"

venv:
	$(CLI) init-venv

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

notebooks:
	jupyter lab
