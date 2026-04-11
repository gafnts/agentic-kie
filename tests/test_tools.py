from __future__ import annotations

from unittest.mock import MagicMock

from agentic_kie.document import PDFDocument
from agentic_kie.tools import create_document_tools


class TestCreateDocumentTools:
    def test_returns_two_tools_by_default(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        assert len(tools) == 2

    def test_returns_three_tools_with_multimodal(
        self, pdf_document: PDFDocument
    ) -> None:
        tools = create_document_tools(pdf_document, modality="multimodal")
        assert len(tools) == 3

    def test_returns_two_tools_with_image(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document, modality="image")
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"get_page_count", "load_images"}

    def test_tool_names_multimodal(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document, modality="multimodal")
        names = {t.name for t in tools}
        assert names == {"get_page_count", "read_text", "load_images"}

    def test_text_only_excludes_load_images(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        names = {t.name for t in tools}
        assert "load_images" not in names


class TestGetPageCount:
    def test_returns_correct_count(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        tool = next(t for t in tools if t.name == "get_page_count")
        assert tool.invoke({}) == 3


class TestReadText:
    def test_returns_single_page_content(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        tool = next(t for t in tools if t.name == "read_text")
        result = tool.invoke({"start": 0})
        assert "Non-Disclosure Agreement" in result

    def test_returns_range_content(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document)
        tool = next(t for t in tools if t.name == "read_text")
        result = tool.invoke({"start": 0, "end": 2})
        assert isinstance(result, str)

    def test_out_of_bounds_returns_error_string(
        self, pdf_document: PDFDocument
    ) -> None:
        tools = create_document_tools(pdf_document)
        tool = next(t for t in tools if t.name == "read_text")
        result = tool.invoke({"start": 99})
        assert isinstance(result, str)
        assert "Error" in result


class TestLoadImages:
    def test_returns_image_url_blocks(
        self, pdf_document: PDFDocument, patched_pymupdf: MagicMock
    ) -> None:
        tools = create_document_tools(pdf_document, modality="multimodal")
        tool = next(t for t in tools if t.name == "load_images")
        result = tool.invoke({"start": 0})
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_returns_range_of_images(
        self, pdf_document: PDFDocument, patched_pymupdf: MagicMock
    ) -> None:
        tools = create_document_tools(pdf_document, modality="multimodal")
        tool = next(t for t in tools if t.name == "load_images")
        result = tool.invoke({"start": 0, "end": 2})
        assert isinstance(result, list)
        assert len(result) == 2

    def test_out_of_bounds_returns_error_list(self, pdf_document: PDFDocument) -> None:
        tools = create_document_tools(pdf_document, modality="multimodal")
        tool = next(t for t in tools if t.name == "load_images")
        result = tool.invoke({"start": 99})
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert "Error" in result[0]["text"]
