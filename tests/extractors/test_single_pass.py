from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agentic_kie.document import PDFDocument
from agentic_kie.extractors.single_pass import SinglePassExtractor
from agentic_kie.prompts import SINGLE_PASS_SYSTEM_PROMPT


class _Schema(BaseModel):
    name: str | None = None
    value: int | None = None


_EXPECTED = _Schema(name="test", value=42)


@pytest.fixture
def mock_chain() -> MagicMock:
    chain = MagicMock()
    chain.invoke.return_value = _EXPECTED
    return chain


@pytest.fixture
def mock_model(mock_chain: MagicMock) -> MagicMock:
    model = MagicMock()
    model.with_structured_output.return_value = mock_chain
    return model


class TestInit:
    def test_binds_schema_to_model(self, mock_model: MagicMock) -> None:
        SinglePassExtractor(model=mock_model, schema=_Schema)
        mock_model.with_structured_output.assert_called_once_with(_Schema)

    def test_uses_default_system_prompt(self, mock_model: MagicMock) -> None:
        extractor = SinglePassExtractor(model=mock_model, schema=_Schema)
        assert extractor._system_prompt == SINGLE_PASS_SYSTEM_PROMPT

    def test_accepts_custom_system_prompt(self, mock_model: MagicMock) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, system_prompt="Custom"
        )
        assert extractor._system_prompt == "Custom"

    def test_defaults_to_text_only(self, mock_model: MagicMock) -> None:
        extractor = SinglePassExtractor(model=mock_model, schema=_Schema)
        assert extractor._modality == "text"


class TestExtract:
    def test_returns_schema_instance(
        self, mock_model: MagicMock, mock_chain: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = SinglePassExtractor(model=mock_model, schema=_Schema)
        result = extractor.extract(pdf_document)
        assert isinstance(result, _Schema)
        assert result.name == "test"
        assert result.value == 42

    def test_invokes_chain_once(
        self, mock_model: MagicMock, mock_chain: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = SinglePassExtractor(model=mock_model, schema=_Schema)
        extractor.extract(pdf_document)
        mock_chain.invoke.assert_called_once()

    def test_passes_system_and_human_messages(
        self, mock_model: MagicMock, mock_chain: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = SinglePassExtractor(model=mock_model, schema=_Schema)
        extractor.extract(pdf_document)
        messages = mock_chain.invoke.call_args[0][0]
        assert len(messages) == 2

    def test_system_message_contains_prompt(
        self, mock_model: MagicMock, mock_chain: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = SinglePassExtractor(model=mock_model, schema=_Schema)
        extractor.extract(pdf_document)
        messages = mock_chain.invoke.call_args[0][0]
        assert messages[0].content == SINGLE_PASS_SYSTEM_PROMPT

    def test_custom_prompt_is_forwarded(
        self, mock_model: MagicMock, mock_chain: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, system_prompt="Custom"
        )
        extractor.extract(pdf_document)
        messages = mock_chain.invoke.call_args[0][0]
        assert messages[0].content == "Custom"


class TestBuildContent:
    def test_text_only_returns_full_text(
        self, mock_model: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = SinglePassExtractor(model=mock_model, schema=_Schema)
        content = extractor._build_content(pdf_document)
        assert isinstance(content, str)
        assert content == pdf_document.full_text

    def test_multimodal_returns_list(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)

    def test_multimodal_text_block_is_first(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        assert isinstance(content[0], dict)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == pdf_document.full_text

    def test_multimodal_includes_one_image_per_page(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        image_blocks = [
            c for c in content if isinstance(c, dict) and c["type"] == "image_url"
        ]
        assert len(image_blocks) == pdf_document.page_count

    def test_multimodal_images_are_base64_data_urls(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        image_blocks = [
            c for c in content if isinstance(c, dict) and c["type"] == "image_url"
        ]
        for block in image_blocks:
            url = block["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")

    def test_multimodal_total_blocks_is_pages_plus_text(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        assert len(content) == pdf_document.page_count + 1

    def test_image_returns_list(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="image"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)

    def test_image_has_no_text_block(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="image"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        text_blocks = [
            c for c in content if isinstance(c, dict) and c.get("type") == "text"
        ]
        assert len(text_blocks) == 0

    def test_image_includes_one_image_per_page(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="image"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        image_blocks = [
            c for c in content if isinstance(c, dict) and c["type"] == "image_url"
        ]
        assert len(image_blocks) == pdf_document.page_count

    def test_image_images_are_base64_data_urls(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="image"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        for block in content:
            assert isinstance(block, dict)
            url = block["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")

    def test_image_total_blocks_equals_page_count(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        patched_pymupdf: MagicMock,
    ) -> None:
        extractor = SinglePassExtractor(
            model=mock_model, schema=_Schema, modality="image"
        )
        content = extractor._build_content(pdf_document)
        assert isinstance(content, list)
        assert len(content) == pdf_document.page_count
