from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agentic_kie.reader import PDFDocument


class TestPageCount:
    def test_returns_number_of_pages(self, pdf_document: PDFDocument) -> None:
        assert pdf_document.page_count == 3

    def test_empty_document_returns_zero(self) -> None:
        assert PDFDocument([], b"").page_count == 0


class TestIsOcr:
    def test_defaults_to_false(self, pdf_document: PDFDocument) -> None:
        assert pdf_document.is_ocr is False

    def test_reflects_constructor_flag(
        self, pdf_text: list[str], pdf_bytes: bytes
    ) -> None:
        assert PDFDocument(pdf_text, pdf_bytes, ocr=True).is_ocr is True


class TestFullText:
    def test_joins_pages_with_double_newline(self, pdf_document: PDFDocument) -> None:
        text = pdf_document.full_text
        assert "\n\n" in text
        assert "This Non-Disclosure Agreement" in text
        assert "Jurisdiction: Delaware" in text

    def test_preserves_page_order(self, pdf_document: PDFDocument) -> None:
        text = pdf_document.full_text
        nda_pos = text.index("Non-Disclosure Agreement")
        terms_pos = text.index("The parties")
        jurisdiction_pos = text.index("Jurisdiction")
        assert nda_pos < terms_pos < jurisdiction_pos

    def test_empty_document_returns_empty_string(self) -> None:
        assert PDFDocument([], b"").full_text == ""


class TestAllImages:
    def test_returns_base64_strings(
        self, pdf_document: PDFDocument, patched_pymupdf: MagicMock
    ) -> None:
        images = pdf_document.all_images
        assert len(images) == 3
        assert all(isinstance(img, str) for img in images)

    def test_empty_document_returns_no_images(self) -> None:
        assert PDFDocument([], b"").all_images == []


class TestReadText:
    def test_returns_single_page_by_default(self, pdf_document: PDFDocument) -> None:
        text = pdf_document.read_text(0)
        assert "Non-Disclosure Agreement" in text
        assert "The parties" not in text

    def test_returns_range_of_pages(self, pdf_document: PDFDocument) -> None:
        text = pdf_document.read_text(0, 2)
        assert "Non-Disclosure Agreement" in text
        assert "The parties" in text
        assert "Jurisdiction" not in text

    def test_equal_start_and_end_returns_empty(self, pdf_document: PDFDocument) -> None:
        assert pdf_document.read_text(1, 1) == ""

    def test_raises_on_negative_index(self, pdf_document: PDFDocument) -> None:
        with pytest.raises(ValueError, match="Negative indices"):
            pdf_document.read_text(-1)

    def test_raises_on_start_greater_than_end(self, pdf_document: PDFDocument) -> None:
        with pytest.raises(ValueError, match="must not be greater"):
            pdf_document.read_text(2, 1)

    def test_raises_on_start_beyond_page_count(self, pdf_document: PDFDocument) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            pdf_document.read_text(10)


class TestLoadImages:
    def test_returns_single_page_by_default(
        self, pdf_document: PDFDocument, patched_pymupdf: MagicMock
    ) -> None:
        images = pdf_document.load_images(1)
        assert len(images) == 1

    def test_returns_range_of_pages(
        self, pdf_document: PDFDocument, patched_pymupdf: MagicMock
    ) -> None:
        images = pdf_document.load_images(0, 2)
        assert len(images) == 2

    def test_applies_default_dpi_scaling(
        self, pdf_text: list[str], pdf_bytes: bytes, patched_pymupdf: MagicMock
    ) -> None:
        doc = PDFDocument(pdf_text, pdf_bytes)
        doc.load_images(0)
        call_args = patched_pymupdf.get_pixmap.call_args
        matrix = call_args.kwargs.get("matrix") or call_args[0][0]
        assert abs(matrix.a - 150 / 72) < 0.01

    def test_applies_custom_dpi_scaling(
        self, pdf_text: list[str], pdf_bytes: bytes, patched_pymupdf: MagicMock
    ) -> None:
        doc = PDFDocument(pdf_text, pdf_bytes, dpi=300)
        doc.load_images(0)
        call_args = patched_pymupdf.get_pixmap.call_args
        matrix = call_args.kwargs.get("matrix") or call_args[0][0]
        assert abs(matrix.a - 300 / 72) < 0.01

    def test_equal_start_and_end_returns_empty(self, pdf_document: PDFDocument) -> None:
        assert pdf_document.load_images(1, 1) == []

    def test_raises_on_negative_index(self, pdf_document: PDFDocument) -> None:
        with pytest.raises(ValueError, match="Negative indices"):
            pdf_document.load_images(-1)

    def test_raises_on_start_greater_than_end(self, pdf_document: PDFDocument) -> None:
        with pytest.raises(ValueError, match="must not be greater"):
            pdf_document.load_images(2, 1)

    def test_raises_on_start_beyond_page_count(self, pdf_document: PDFDocument) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            pdf_document.load_images(10)
