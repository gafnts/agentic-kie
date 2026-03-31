from __future__ import annotations

from unittest.mock import MagicMock

from agentic_kie.document import PDFDocument
from agentic_kie.tools import create_document_tools


class TestCreateDocumentTools:
    def test_returns_two_tools_by_default(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        assert len(tools) == 2

    def test_returns_three_tools_with_images(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document, include_images=True)
        assert len(tools) == 3

    def test_tool_names(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document, include_images=True)
        names = {t.name for t in tools}
        assert names == {"get_page_count", "read_text", "load_image"}

    def test_text_only_excludes_load_image(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        names = {t.name for t in tools}
        assert "load_image" not in names


class TestGetPageCount:
    def test_returns_correct_count(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        tool = next(t for t in tools if t.name == "get_page_count")
        assert tool.invoke({}) == 3


class TestReadText:
    def test_returns_page_content(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        tool = next(t for t in tools if t.name == "read_text")
        result = tool.invoke({"page_number": 0})
        assert "Non-Disclosure Agreement" in result

    def test_out_of_bounds_returns_error_string(
        self, pdf_document: PDFDocument
    ) -> None:
        tools = create_document_tools(pdf_document)
        tool = next(t for t in tools if t.name == "read_text")
        result = tool.invoke({"page_number": 99})
        assert isinstance(result, str)
        assert "Error" in result


class TestLoadImage:
    def test_returns_image_content_blocks(
        self, pdf_document: PDFDocument, patched_pymupdf: MagicMock
    ) -> None:
        tools = create_document_tools(pdf_document, include_images=True)
        tool = next(t for t in tools if t.name == "load_image")
        result = tool.invoke({"page_number": 0})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_out_of_bounds_returns_error_block(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document, include_images=True)
        tool = next(t for t in tools if t.name == "load_image")
        result = tool.invoke({"page_number": 99})
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert "Error" in result[0]["text"]
