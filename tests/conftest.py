from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from agentic_kie.reader import PDFReader

SAMPLE_METADATA: dict[str, Any] = {
    "producer": "Skia/PDF m77",
    "creator": "Chromium",
    "creationdate": "2019-07-22T13:07:27+00:00",
    "source": "test.pdf",
    "file_path": "test.pdf",
    "total_pages": 3,
    "format": "PDF 1.4",
    "title": "",
    "author": "",
    "subject": "",
    "keywords": "",
    "moddate": "2019-07-22T13:07:27+00:00",
    "trapped": "",
    "modDate": "D:20190722130727+00'00'",
    "creationDate": "D:20190722130727+00'00'",
    "page": 0,
}


@pytest.fixture
def mock_lc_pages() -> list[Document]:
    """Three-page document simulating PyMuPDFLoader output."""
    return [
        Document(
            page_content="This Non-Disclosure Agreement is entered into...",
            metadata={**SAMPLE_METADATA, "page": 0},
        ),
        Document(
            page_content="The parties agree to the following terms...",
            metadata={**SAMPLE_METADATA, "page": 1},
        ),
        Document(
            page_content="Jurisdiction: Delaware. Effective date: 2024-01-15.",
            metadata={**SAMPLE_METADATA, "page": 2},
        ),
    ]


@pytest.fixture
def empty_lc_pages() -> list[Document]:
    """Blank document."""
    return [
        Document(
            page_content="",
            metadata={**SAMPLE_METADATA, "page": 0},
        ),
        Document(
            page_content="",
            metadata={**SAMPLE_METADATA, "page": 1},
        ),
        Document(
            page_content="",
            metadata={**SAMPLE_METADATA, "page": 2},
        ),
    ]


@pytest.fixture
def mock_pymupdf_page() -> MagicMock:
    """A pymupdf Page with a fake pixmap."""
    page = MagicMock()
    pixmap = MagicMock()
    pixmap.tobytes.return_value = b"fake-png-bytes"
    page.get_pixmap.return_value = pixmap
    return page


@pytest.fixture
def pdf_reader(mock_lc_pages: list[Document]) -> PDFReader:
    """PDFReader with patched PyMuPDFLoader."""
    with patch("agentic_kie.reader.PyMuPDFLoader") as mock_loader:
        mock_loader.return_value.load.return_value = mock_lc_pages
        reader = PDFReader("test.pdf")
    return reader
