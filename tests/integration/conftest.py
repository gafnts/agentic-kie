from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests when INTEGRATION_API_KEY is not set."""
    if os.environ.get("INTEGRATION_API_KEY"):
        return
    skip = pytest.mark.skip(reason="INTEGRATION_API_KEY not set")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def sample_pdf_path() -> Path:
    path = FIXTURE_DIR / "sample.pdf"
    assert path.exists(), f"Fixture not found: {path}"
    return path
