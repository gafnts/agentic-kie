# Agentic KIE: LLM-Based Key Information Extraction from Documents

[![CI](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml/badge.svg)](https://github.com/gafnts/agentic-kie/actions/workflows/ci.yml)
[![CD](https://github.com/gafnts/agentic-kie/actions/workflows/cd.yml/badge.svg)](https://github.com/gafnts/agentic-kie/actions/workflows/cd.yml)
[![codecov](https://codecov.io/github/gafnts/agentic-kie/graph/badge.svg)](https://codecov.io/github/gafnts/agentic-kie)
[![PyPI](https://img.shields.io/pypi/v/agentic-kie)](https://pypi.org/project/agentic-kie/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Python package for agentic and single-pass Key Information Extraction (KIE) from documents using large language models.

The package exposes two extraction strategies: a **single-pass** approach that issues one structured prompt per document and parses the response directly against a Pydantic schema, and an **agentic** approach that orchestrates a [LangChain](https://python.langchain.com/)-powered agent loop capable of iterative reasoning, tool use, and multi-step refinement over the document content.

## Contents

- [Installation](#installation)
- [Development](#development)
  - [Linting and formatting](#linting-and-formatting)
  - [Testing](#testing)
- [CI pipeline](#ci-pipeline)
- [Releasing](#releasing)

---

## Installation

The package requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

### As a dependency

Install from [PyPI](https://pypi.org/project/agentic-kie/):

```bash
uv add agentic-kie
```

Or add it directly from GitHub to track the latest unreleased changes:

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

`make install` syncs all dependency groups and installs both the `pre-commit` and `pre-push` hooks.

To install without development dependencies:

```bash
uv sync --no-dev
```

---

## Development

The following `make` targets are available:

| Target | Description |
|---|---|
| `make check` | Run the full pre-commit suite (lint, format, type check) |
| `make lint` | Run `ruff check` on `src` and `tests` |
| `make format` | Run `ruff check --fix` on `src` and `tests` |
| `make type` | Run `mypy` on `src` and `tests` |
| `make test` | Run `pytest` with branch coverage |

### Linting and formatting

```bash
make check
```

### Testing

```bash
make test
```

Coverage is enforced at 95% and is reported to [Codecov](https://codecov.io/github/gafnts/agentic-kie) on every CI run.

---

## CI pipeline

GitHub Actions runs two sequential jobs on every push and pull request to `main`:

1. **`lint-and-type-check`**: runs `ruff check`, `ruff format --check`, and `mypy`.
2. **`test`**: runs `pytest` with branch coverage and uploads the `coverage.xml` report to Codecov.

---

## CD pipeline

### Releasing

Releases are driven by git tags. The package version is derived automatically from the tag via [hatch-vcs](https://github.com/ofek/hatch-vcs), so no manual version bumps are needed.

To cut a new release, push a version tag from `main`:

```bash
git tag v1.2.3
git push origin v1.2.3
```

This triggers the CD pipeline, which runs the following jobs:

1. **`guard`**: verifies the tag points to a commit on `main`, blocking releases from feature branches.
2. **`lint-and-type-check`** and **`test`** (parallel, both require `guard`): same checks as CI.
3. **`publish`** (requires both above): builds the package with `uv build`, publishes to [PyPI](https://pypi.org/project/agentic-kie/) via trusted publishing, and creates a GitHub Release with auto-generated notes and the built distribution files attached.
