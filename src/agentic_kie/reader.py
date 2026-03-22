from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from pymupdf import pymupdf


class PDFReader:
    """
    `PDFReader` is a high-level orchestrator that gives both extraction modes
    (single-pass and agentic) a unified interface to read a PDF.

    Responsabilities:

        1. Load and hold content at both document and page granularity
        2. Detect empty or broken documents
        3. Fall back to Textract OCR
        4. Provide and format images
        5. Expose a reading interface for the agent tools
        6. Serve both extraction modes transparently
    """

    def __init__(self, path: Path | str, *, dpi: int = 150) -> None:
        self._path = Path(path)
        self._dpi = dpi
        self._pages: list[Document] = PyMuPDFLoader(path, mode="page").load()

    def get_full_text(self) -> str:
        return "\n\n".join([page.page_content for page in self._pages])

    def get_all_images(self) -> list[str]:
        document = pymupdf.open(self._path)  # type: ignore[no-untyped-call]
        return [self._page_to_png(page, self._dpi) for page in document]  # type: ignore[attr-defined]

    @staticmethod
    def _page_to_png(page: pymupdf.Page, dpi: int) -> str:
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)  # type: ignore[no-untyped-call]
        pixel_grid = page.get_pixmap(matrix=matrix)
        return base64.b64encode(pixel_grid.tobytes("png")).decode()  # type: ignore[no-untyped-call]

    @property
    def page_count(self) -> int:
        return len(self._pages)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._pages[0].metadata if self._pages else {}
