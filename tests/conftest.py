from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentic_kie.reader import PDFDocument

SAMPLE_TEXT: list[str] = [
    "This Non-Disclosure Agreement is entered into...",
    "The parties agree to the following terms...",
    "Jurisdiction: Delaware. Effective date: 2024-01-15.",
]

SAMPLE_BYTES = b"%PDF-1.4 fake-pdf-bytes"


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
