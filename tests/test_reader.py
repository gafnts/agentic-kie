from __future__ import annotations

from unittest.mock import MagicMock, patch

from agentic_kie.reader import PDFReader


class TestPageCount:
    def test_returns_number_of_pages(self, pdf_reader: PDFReader) -> None:
        assert pdf_reader.page_count == 3


class TestMetadata:
    def test_returns_first_page_metadata(self, pdf_reader: PDFReader) -> None:
        meta = pdf_reader.metadata
        assert meta["producer"] == "Skia/PDF m77"
        assert meta["format"] == "PDF 1.4"
        assert meta["page"] == 0

    def test_returns_empty_dict_when_no_pages(self) -> None:
        with patch("agentic_kie.reader.PyMuPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = []
            reader = PDFReader("empty.pdf")
        assert reader.metadata == {}


class TestGetFullText:
    def test_joins_pages_with_double_newline(self, pdf_reader: PDFReader) -> None:
        text = pdf_reader.get_full_text()
        assert "\n\n" in text
        assert "This Non-Disclosure Agreement" in text
        assert "Jurisdiction: Delaware" in text

    def test_preserves_page_order(self, pdf_reader: PDFReader) -> None:
        text = pdf_reader.get_full_text()
        nda_pos = text.index("Non-Disclosure Agreement")
        terms_pos = text.index("parties agree")
        jurisdiction_pos = text.index("Jurisdiction")
        assert nda_pos < terms_pos < jurisdiction_pos


class TestGetAllImages:
    def test_returns_base64_strings(
        self, pdf_reader: PDFReader, mock_pymupdf_page: MagicMock
    ) -> None:
        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_pymupdf_page] * 3)

        with patch("agentic_kie.reader.pymupdf.open", return_value=mock_doc):
            images = pdf_reader.get_all_images()

        assert len(images) == 3
        assert all(isinstance(img, str) for img in images)

    def test_applies_dpi_scaling(self, mock_pymupdf_page: MagicMock) -> None:
        with patch("agentic_kie.reader.PyMuPDFLoader") as mock_loader:
            mock_loader.return_value.load.return_value = []
            reader = PDFReader("test.pdf", dpi=300)

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_pymupdf_page])

        with patch("agentic_kie.reader.pymupdf.open", return_value=mock_doc):
            reader.get_all_images()

        call_args = mock_pymupdf_page.get_pixmap.call_args
        matrix = call_args.kwargs.get("matrix") or call_args[0][0]
        # 300 / 72 ≈ 4.17
        assert abs(matrix.a - 300 / 72) < 0.01
