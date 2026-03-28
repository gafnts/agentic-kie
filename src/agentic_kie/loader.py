from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pymupdf

from agentic_kie.document import PDFDocument
from agentic_kie.exceptions import (
    CorruptDocumentError,
    EmptyDocumentError,
    OCRNotConfiguredError,
    PasswordProtectedError,
)
from agentic_kie.ocr import OCRProvider

logger = logging.getLogger(__name__)


# Minimum average characters per page to consider a text layer present.
_DEFAULT_TEXT_THRESHOLD: int = 32


class PDFLoader:
    """
    Ingestion boundary between raw PDF files and clean document representations.

    This class absorbs real-world PDF complexity — file I/O, text-layer
    detection, OCR routing, and error handling — so that PDFDocument can
    remain a clean, agent-facing representation. Every PDFDocument returned
    is guaranteed to be validated and usable.

    The separation of concerns is intentional: callers interact with a
    clean document interface without awareness of how it was loaded or
    whether OCR was needed.

    Parameters
    ----------
    ocr_provider:
        Strategy for extracting text from scanned pages.
        Required only if you expect to process scanned documents.
    dpi:
        Resolution for rendering pages to images (used during OCR).
    text_threshold:
        Minimum average characters per page to consider a
        text layer present. Below this, the document is treated
        as scanned.
    """

    def __init__(
        self,
        ocr_provider: OCRProvider | None = None,
        dpi: int = 150,
        text_threshold: int = _DEFAULT_TEXT_THRESHOLD,
    ) -> None:
        self._ocr_provider = ocr_provider
        self._dpi = dpi
        self._text_threshold = text_threshold

    def load(self, path: Path) -> PDFDocument:
        """
        Load and validate a PDF, returning a clean PDFDocument.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        CorruptDocumentError
            If the file cannot be parsed as a PDF.
        PasswordProtectedError
            If the PDF is encrypted.
        EmptyDocumentError
            If the document has zero pages, or yields no text
            after exhausting all extraction options.
        OCRNotConfiguredError
            If the document has no text layer and no OCR provider
            was configured.
        """
        self._validate_path(path)
        doc = self._open(path)

        try:
            self._validate_structure(doc)
            pdf_bytes = path.read_bytes()

            if self._has_text_layer(doc):
                logger.info("Text layer detected in '%s'", path.name)
                text_pages = [
                    cast(str, doc[i].get_text("text")).strip()  # type: ignore[no-untyped-call]
                    for i in range(doc.page_count)
                ]
                return PDFDocument(text_pages, pdf_bytes, dpi=self._dpi)

            logger.info("No text layer in '%s', routing to OCR", path.name)

            ocr_text = self._run_ocr(doc, path)
            text_pages = [ocr_text.get(i, "") for i in range(doc.page_count)]

            return PDFDocument(text_pages, pdf_bytes, dpi=self._dpi, ocr=True)

        finally:
            doc.close()  # type: ignore[no-untyped-call]

    @staticmethod
    def _validate_path(path: Path) -> None:
        """Raise FileNotFoundError if path does not point to an existing file."""
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

    @staticmethod
    def _open(path: Path) -> pymupdf.Document:
        """
        Open a PDF file and return the parsed document.

        Raises
        ------
        CorruptDocumentError
            If the file cannot be parsed as a PDF.
        PasswordProtectedError
            If the PDF is encrypted.
        """
        try:
            doc = pymupdf.open(path)  # type: ignore[no-untyped-call]
        except pymupdf.FileDataError as e:
            raise CorruptDocumentError(f"Cannot parse '{path.name}' as PDF: {e}") from e

        if doc.is_encrypted:
            doc.close()  # type: ignore[no-untyped-call]
            raise PasswordProtectedError(f"'{path.name}' is password-protected")

        return doc

    @staticmethod
    def _validate_structure(doc: pymupdf.Document) -> None:
        """Raise EmptyDocumentError if the document has zero pages."""
        if doc.page_count == 0:
            raise EmptyDocumentError("Document has zero pages")

    def _has_text_layer(self, doc: pymupdf.Document) -> bool:
        """
        Check whether the document has a meaningful text layer.

        Uses a simple heuristic: average characters per page must
        exceed the configured threshold.
        """
        total_chars = sum(
            len(cast(str, doc[i].get_text("text")).strip())  # type: ignore[no-untyped-call, misc]
            for i in range(doc.page_count)
        )
        avg_chars = total_chars / doc.page_count
        logger.debug(
            "Text layer check: %d total chars, %.1f avg/page (threshold: %d)",
            total_chars,
            avg_chars,
            self._text_threshold,
        )
        return bool(avg_chars >= self._text_threshold)

    def _run_ocr(self, doc: pymupdf.Document, path: Path) -> dict[int, str]:
        """
        OCR all pages, returning a page-number → text mapping.

        Raises
        ------
        OCRNotConfiguredError
            If no OCR provider is configured.
        EmptyDocumentError
            If OCR produces no text for any page.
        """
        if self._ocr_provider is None:
            raise OCRNotConfiguredError(
                f"'{path.name}' appears scanned but no OCR provider "
                "was configured. Pass an OCRProvider to PDFLoader."
            )

        ocr_text: dict[int, str] = {}

        for page_num in range(doc.page_count):
            page = doc[page_num]
            pixmap = page.get_pixmap(dpi=self._dpi)
            image_bytes = pixmap.tobytes("png")  # type: ignore[no-untyped-call]

            text = self._ocr_provider.extract_text(image_bytes).strip()
            if text:
                ocr_text[page_num] = text

        if not ocr_text:
            raise EmptyDocumentError(f"OCR produced no text for '{path.name}'")

        logger.info(
            "OCR extracted text from %d/%d pages of '%s'",
            len(ocr_text),
            doc.page_count,
            path.name,
        )

        return ocr_text
