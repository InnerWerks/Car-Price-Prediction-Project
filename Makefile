.PHONY: help venv shell download redownload download-dataset prepare data train evaluate notebooks

# Defaults
PY ?= python
CLI := $(PY) -m src.cli
CONFIG ?= configs

help:
	@echo "Available targets:"
	@echo "  venv              Create .venv and install requirements"
	@echo "  shell             Create/activate venv and open an activated shell"
	@echo "  download          Download dataset (uses configs/dataset.yaml)"
	@echo "  redownload        Force re-download dataset (overwrite)"
	@echo "  download-dataset  Download with custom slug (set DATASET=owner/slug)"
	@echo "  prepare           Create data/ and models/ directories"
	@echo "  data              Build datasets: load/split/preprocess -> data/*"
	@echo "  train             Run training (stub)"
	@echo "  evaluate          Run evaluation (stub)"
	@echo "  notebooks         Launch Jupyter Lab"

venv:
	$(CLI) init-venv

shell:
	$(CLI) init-venv --shell

download:
	$(CLI) download-data --config $(CONFIG)

redownload:
	$(CLI) download-data --config $(CONFIG) --force

download-dataset:
	@test -n "$(DATASET)" || (echo "Set DATASET=owner/slug (e.g., bhavikjikadara/car-price-prediction-dataset)" && false)
	$(CLI) download-data --config $(CONFIG) --dataset $(DATASET)

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
