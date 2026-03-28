from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_kie.document import PDFDocument
from agentic_kie.exceptions import (
    CorruptDocumentError,
    EmptyDocumentError,
    OCRNotConfiguredError,
    PasswordProtectedError,
)
from agentic_kie.loader import PDFLoader


def _make_page(text: str = "") -> MagicMock:
    page = MagicMock()
    page.get_text.return_value = text
    pixmap = MagicMock()
    pixmap.tobytes.return_value = b"fake-png-bytes"
    page.get_pixmap.return_value = pixmap
    return page


def _make_doc(
    page_count: int = 2,
    *,
    is_encrypted: bool = False,
    chars_per_page: int = 100,
) -> tuple[MagicMock, MagicMock]:
    page = _make_page("x" * chars_per_page)
    doc = MagicMock()
    doc.is_encrypted = is_encrypted
    doc.page_count = page_count
    doc.__getitem__ = lambda self, i: page
    return doc, page


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return p


@pytest.fixture
def mock_ocr_provider() -> MagicMock:
    provider = MagicMock()
    provider.extract_text.return_value = "OCR extracted text"
    return provider


class TestLoad:
    def test_text_pdf_returns_pdf_document(self, pdf_path: Path) -> None:
        doc, _ = _make_doc(chars_per_page=100)
        with patch("agentic_kie.loader.pymupdf.open", return_value=doc):
            result = PDFLoader().load(pdf_path)
        assert isinstance(result, PDFDocument)

    def test_text_pdf_is_not_ocr(self, pdf_path: Path) -> None:
        doc, _ = _make_doc(chars_per_page=100)
        with patch("agentic_kie.loader.pymupdf.open", return_value=doc):
            result = PDFLoader().load(pdf_path)
        assert result.is_ocr is False

    def test_text_pdf_page_count_matches_document(self, pdf_path: Path) -> None:
        doc, _ = _make_doc(page_count=3, chars_per_page=100)
        with patch("agentic_kie.loader.pymupdf.open", return_value=doc):
            result = PDFLoader().load(pdf_path)
        assert result.page_count == 3

    def test_scanned_pdf_sets_ocr_flag(
        self, pdf_path: Path, mock_ocr_provider: MagicMock
    ) -> None:
        doc, _ = _make_doc(chars_per_page=0)
        with patch("agentic_kie.loader.pymupdf.open", return_value=doc):
            result = PDFLoader(ocr_provider=mock_ocr_provider).load(pdf_path)
        assert result.is_ocr is True

    def test_scanned_pdf_calls_ocr_per_page(
        self, pdf_path: Path, mock_ocr_provider: MagicMock
    ) -> None:
        doc, _ = _make_doc(page_count=3, chars_per_page=0)
        with patch("agentic_kie.loader.pymupdf.open", return_value=doc):
            PDFLoader(ocr_provider=mock_ocr_provider).load(pdf_path)
        assert mock_ocr_provider.extract_text.call_count == 3

    def test_doc_is_closed_after_successful_load(self, pdf_path: Path) -> None:
        doc, _ = _make_doc(chars_per_page=100)
        with patch("agentic_kie.loader.pymupdf.open", return_value=doc):
            PDFLoader().load(pdf_path)
        doc.close.assert_called_once()

    def test_doc_is_closed_when_error_is_raised(self, pdf_path: Path) -> None:
        doc, _ = _make_doc(page_count=0)
        with (
            patch("agentic_kie.loader.pymupdf.open", return_value=doc),
            pytest.raises(EmptyDocumentError),
        ):
            PDFLoader().load(pdf_path)
        doc.close.assert_called_once()

    def test_raises_file_not_found_for_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            PDFLoader().load(tmp_path / "nonexistent.pdf")

    def test_raises_corrupt_document_on_bad_pdf(self, pdf_path: Path) -> None:
        with (
            patch("agentic_kie.loader.pymupdf.open", side_effect=Exception("bad file")),
            pytest.raises(CorruptDocumentError),
        ):
            PDFLoader().load(pdf_path)

    def test_raises_password_protected_on_encrypted_pdf(self, pdf_path: Path) -> None:
        doc, _ = _make_doc(is_encrypted=True)
        with (
            patch("agentic_kie.loader.pymupdf.open", return_value=doc),
            pytest.raises(PasswordProtectedError),
        ):
            PDFLoader().load(pdf_path)

    def test_raises_empty_document_on_zero_pages(self, pdf_path: Path) -> None:
        doc, _ = _make_doc(page_count=0)
        with (
            patch("agentic_kie.loader.pymupdf.open", return_value=doc),
            pytest.raises(EmptyDocumentError, match="zero pages"),
        ):
            PDFLoader().load(pdf_path)

    def test_raises_ocr_not_configured_when_scanned_without_provider(
        self, pdf_path: Path
    ) -> None:
        doc, _ = _make_doc(chars_per_page=0)
        with (
            patch("agentic_kie.loader.pymupdf.open", return_value=doc),
            pytest.raises(OCRNotConfiguredError),
        ):
            PDFLoader().load(pdf_path)

    def test_raises_empty_document_when_ocr_yields_no_text(
        self, pdf_path: Path, mock_ocr_provider: MagicMock
    ) -> None:
        mock_ocr_provider.extract_text.return_value = ""
        doc, _ = _make_doc(chars_per_page=0)
        with (
            patch("agentic_kie.loader.pymupdf.open", return_value=doc),
            pytest.raises(EmptyDocumentError),
        ):
            PDFLoader(ocr_provider=mock_ocr_provider).load(pdf_path)


class TestHasTextLayer:
    def test_returns_true_when_avg_exceeds_threshold(self) -> None:
        doc, _ = _make_doc(page_count=1, chars_per_page=100)
        assert PDFLoader()._has_text_layer(doc) is True

    def test_returns_false_when_avg_below_threshold(self) -> None:
        doc, _ = _make_doc(page_count=1, chars_per_page=0)
        assert PDFLoader()._has_text_layer(doc) is False

    def test_uses_configured_threshold(self) -> None:
        doc, _ = _make_doc(page_count=1, chars_per_page=10)
        assert PDFLoader(text_threshold=5)._has_text_layer(doc) is True
        assert PDFLoader(text_threshold=20)._has_text_layer(doc) is False

    def test_averages_across_all_pages(self) -> None:
        # Page 0: 100 chars, page 1: 0 chars → avg 50 → above default threshold of 32
        pages = [_make_page("x" * 100), _make_page("")]
        doc = MagicMock()
        doc.page_count = 2
        doc.__getitem__ = lambda self, i: pages[i]
        assert PDFLoader()._has_text_layer(doc) is True


class TestRunOcr:
    def test_returns_page_text_mapping(
        self, pdf_path: Path, mock_ocr_provider: MagicMock
    ) -> None:
        doc, _ = _make_doc(page_count=2, chars_per_page=0)
        result = PDFLoader(ocr_provider=mock_ocr_provider)._run_ocr(doc, pdf_path)
        assert result == {0: "OCR extracted text", 1: "OCR extracted text"}

    def test_skips_pages_with_no_ocr_text(
        self, pdf_path: Path, mock_ocr_provider: MagicMock
    ) -> None:
        mock_ocr_provider.extract_text.side_effect = ["text on page 0", ""]
        doc, _ = _make_doc(page_count=2, chars_per_page=0)
        result = PDFLoader(ocr_provider=mock_ocr_provider)._run_ocr(doc, pdf_path)
        assert result == {0: "text on page 0"}
        assert 1 not in result

    def test_passes_dpi_to_get_pixmap(
        self, pdf_path: Path, mock_ocr_provider: MagicMock
    ) -> None:
        doc, page = _make_doc(page_count=1, chars_per_page=0)
        PDFLoader(ocr_provider=mock_ocr_provider, dpi=300)._run_ocr(doc, pdf_path)
        page.get_pixmap.assert_called_once_with(dpi=300)

    def test_raises_when_no_provider(self, pdf_path: Path) -> None:
        doc, _ = _make_doc()
        with pytest.raises(OCRNotConfiguredError):
            PDFLoader()._run_ocr(doc, pdf_path)

    def test_raises_empty_document_when_all_pages_empty(
        self, pdf_path: Path, mock_ocr_provider: MagicMock
    ) -> None:
        mock_ocr_provider.extract_text.return_value = ""
        doc, _ = _make_doc(page_count=2, chars_per_page=0)
        with pytest.raises(EmptyDocumentError):
            PDFLoader(ocr_provider=mock_ocr_provider)._run_ocr(doc, pdf_path)
