# web-graph-embeddings — reproduction pipeline for the Common Crawl host embeddings.

.DEFAULT_GOAL := help
SHELL := /bin/bash

VARIANT ?= cpu
IMAGE   ?= web-graph-embeddings-$(VARIANT):latest

.PHONY: help
help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Create the venv and install the project + dev extras.
	uv venv --python 3.12
	uv sync --extra dev

.PHONY: lint
lint: ## Run ruff lint checks.
	uv run ruff check src tests

.PHONY: format
format: ## Auto-format with ruff.
	uv run ruff format src tests
	uv run ruff check --fix src tests

.PHONY: format-check
format-check: ## Check formatting without modifying files.
	uv run ruff format --check src tests

.PHONY: check
check: lint format-check ## Lint + format check, no mutation.

.PHONY: test
test: ## Run the unit test suite.
	uv run pytest

.PHONY: image
image: ## Build a container image (VARIANT=cpu|cugraph-pyg).
	docker build -f docker/$(VARIANT)/Dockerfile -t $(IMAGE) .
