# Contributing

## Development setup

The package requires Python 3.13 or later. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

Clone the repository and install all dependencies including dev tools and git hooks:

```bash
git clone https://github.com/gafnts/agentic-kie.git
cd agentic-kie
make install
```

To install without development dependencies:

```bash
uv sync --no-dev
```

## Available targets

| Target | Description |
|---|---|
| `make check` | Run the full pre-commit suite (lint, format, type check) |
| `make lint` | Run `ruff check` on `src` and `tests` |
| `make format` | Run `ruff check --fix` on `src` and `tests` |
| `make type` | Run `mypy` on `src` and `tests` |
| `make test` | Run `pytest` with branch coverage |

## CI pipeline

GitHub Actions runs two sequential jobs on every push and pull request to `main`:

1. **`lint-and-type-check`**: runs `ruff check`, `ruff format --check`, and `mypy`.
2. **`test`**: runs `pytest` with branch coverage and uploads the `coverage.xml` report to Codecov.

Coverage is enforced at 95%.

## Releasing

Releases are driven by git tags. The package version is derived automatically from the tag via [hatch-vcs](https://github.com/ofek/hatch-vcs), so no manual version bumps are needed.

To cut a new release, push a version tag from `main`:

```bash
git tag v1.2.3
git push origin v1.2.3
```

This triggers the CD pipeline:

1. **`guard`**: verifies the tag points to a commit on `main`.
2. **`lint-and-type-check`** and **`test`** (parallel): same checks as CI.
3. **`publish`**: builds with `uv build`, publishes to [PyPI](https://pypi.org/project/agentic-kie/) via trusted publishing, and creates a GitHub Release with the built distribution files attached.
