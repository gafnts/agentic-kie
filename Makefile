.DEFAULT_GOAL := help

.PHONY: help install \
        check lint format type \
        test integration \
        langgraph

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'


# LOCAL DEVELOPMENT SETUP

install: ## Sync deps and install pre-commit hooks (both stages)
	uv sync --all-groups --all-extras
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push


# QUALITY GATES

check: ## Run every pre-commit hook against every file (both stages)
	uv run pre-commit run --all-files --hook-stage pre-commit
	uv run pre-commit run --all-files --hook-stage pre-push

lint: ## Run ruff check on src and tests
	uv run ruff check src tests

format: ## Apply ruff lint fixes and formatting to src and tests
	uv run ruff check src tests --fix
	uv run ruff format src tests

type: ## Run mypy on src and tests
	uv run mypy src tests


# TESTING

test: ## Run pytest with branch coverage
	uv run pytest --cov --cov-branch --cov-report=xml -v

integration: ## Run integration-marked tests
	uv run pytest -m "integration" -v


# DEVELOPMENT

langgraph: ## Start the LangGraph dev server
	uv run langgraph dev
