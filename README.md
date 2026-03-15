# Agentic KIE: LLM-Based Key Information Extraction from Documents

[![CI](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml/badge.svg)](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/gafnts/agentic-kie/graph/badge.svg)](https://codecov.io/github/gafnts/agentic-kie)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Python package for agentic and single-pass Key Information Extraction (KIE) from documents using large language models.

The package exposes two extraction strategies: a **single-pass** approach that issues one structured prompt per document and parses the response directly against a Pydantic schema, and an **agentic** approach that orchestrates a [LangChain](https://python.langchain.com/)-powered agent loop capable of iterative reasoning, tool use, and multi-step refinement over the document content.

## Contents

- [Installation](#installation)
- [Development](#development)
  - [Linting and formatting](#linting-and-formatting)
  - [Testing](#testing)
- [CI pipeline](#ci-pipeline)

---

## Installation

The package requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

### As a dependency

To use this package in your own project without cloning the repository, add it directly from GitHub:

```bash
uv add git+https://github.com/gafnts/agentic-kie
```

### For development

Clone the repository and install the package with all dependencies (including dev tools):

```bash
git clone https://github.com/gafnts/agentic-kie.git
cd agentic-kie
make install
```

Use `make install` to sync dependencies and set up pre-commit hooks (both `pre-commit` and `pre-push`).

To install without development dependencies:

```bash
uv sync --no-dev
```

---

## Development

### Linting and formatting

```bash
# Run the linting, formatting and type checking suite
uv run pre-commit run --all-files
```

### Testing

```bash
# Run the test suite with coverage
uv run pytest --cov --cov-branch --cov-report=term-missing
```

Coverage is enforced at 95% and is reported to [Codecov](https://codecov.io/github/gafnts/agentic-kie) on every CI run.

---

## CI pipeline

GitHub Actions runs two sequential jobs on every push and pull request to `main`:

1. **`lint-and-type-check`**: runs `ruff check`, `ruff format --check`, and `mypy`.
2. **`test`**: runs `pytest` with branch coverage and uploads the `coverage.xml` report to Codecov.
