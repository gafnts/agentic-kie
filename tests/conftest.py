from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from agentic_kie.document import PDFDocument

SAMPLE_TEXT: list[str] = [
    "This Non-Disclosure Agreement is entered into...",
    "The parties agree to the following terms...",
    "Jurisdiction: Delaware. Effective date: 2024-01-15.",
]

SAMPLE_BYTES: bytes = b"%PDF-1.4 fake-pdf-bytes"


@pytest.fixture
def pdf_text() -> list[str]:
    return SAMPLE_TEXT


@pytest.fixture
def pdf_bytes() -> bytes:
    return SAMPLE_BYTES


@pytest.fixture
def pdf_document(pdf_text: list[str], pdf_bytes: bytes) -> PDFDocument:
    """Three-page PDFDocument with sample NDA content."""
    return PDFDocument(pdf_text, pdf_bytes)


@pytest.fixture
def mock_pymupdf_page() -> MagicMock:
    """A pymupdf Page with a fake pixmap."""
    page = MagicMock()
    pixmap = MagicMock()
    pixmap.tobytes.return_value = b"fake-png-bytes"
    page.get_pixmap.return_value = pixmap
    return page


@pytest.fixture
def patched_pymupdf(mock_pymupdf_page: MagicMock) -> Generator[MagicMock]:
    """Patches pymupdf.open and yields the page mock for call inspection."""
    mock_doc = MagicMock()
    mock_doc.__getitem__ = lambda self, i: mock_pymupdf_page
    mock_doc.__enter__ = lambda self: mock_doc
    with patch("agentic_kie.document.pymupdf.open", return_value=mock_doc):
        yield mock_pymupdf_page
