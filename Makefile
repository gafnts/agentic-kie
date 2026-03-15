.PHONY: install

install:
	uv sync
	uv run pre-commit install
	uv run pre-commit install --hook-type pre-push
