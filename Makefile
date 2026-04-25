.PHONY: help install stimuli pilot sandbag probes music figures test lint clean

help:
	@echo "Targets:"
	@echo "  install   - uv sync core + dev deps"
	@echo "  install-ml- uv sync with torch/transformers/vllm"
	@echo "  stimuli   - Phase 1: generate stratified stimulus quadruplets"
	@echo "  pilot     - Phase 2: run pairwise preference elicitation"
	@echo "  sandbag   - Phase 3: run sandbagging conditions + MMLU/HumanEval"
	@echo "  probes    - Phase 4: extract activations + train linear probes"
	@echo "  music     - Phase 5: run music generalization"
	@echo "  figures   - Regenerate all paper figures"
	@echo "  test      - pytest"
	@echo "  lint      - ruff check + format check"

install:
	uv sync --extra dev

install-ml:
	uv sync --extra dev --extra ml

stimuli:
	uv run python -m sandbagging_defense.stimuli.generators \
		--out results/stimuli/v1.parquet --n 2000 --seed 0

pilot:
	uv run python -m sandbagging_defense.elicitation.harness \
		--config configs/pilot.yaml

sandbag:
	uv run python -m sandbagging_defense.sandbagging.run \
		--config configs/sandbag.yaml

probes:
	uv run python -m sandbagging_defense.probes.train \
		--config configs/probes.yaml

music:
	uv run python -m sandbagging_defense.stimuli.music \
		--out results/music/v1.parquet

figures:
	uv run python -m sandbagging_defense.analysis.figures \
		--out paper/figures/

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
