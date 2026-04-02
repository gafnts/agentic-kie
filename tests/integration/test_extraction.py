from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel

from agentic_kie import AgenticExtractor, PDFLoader, SinglePassExtractor

from .schema import SimpleDocument

MODEL_ID = "claude-haiku-4-5"


def _make_model() -> BaseChatModel:
    """Build a chat model using the integration API key."""
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=MODEL_ID,  # type: ignore[call-arg]
        api_key=os.environ["INTEGRATION_API_KEY"],
    )


@pytest.mark.integration
class TestSinglePass:
    def test_text_returns_valid_schema(self, sample_pdf_path: Path) -> None:
        doc = PDFLoader().load(sample_pdf_path)
        extractor = SinglePassExtractor(
            model=_make_model(),
            schema=SimpleDocument,
            modality="text",
        )
        result = extractor.extract(doc)
        assert isinstance(result, SimpleDocument)

    def test_multimodal_returns_valid_schema(self, sample_pdf_path: Path) -> None:
        doc = PDFLoader().load(sample_pdf_path)
        extractor = SinglePassExtractor(
            model=_make_model(),
            schema=SimpleDocument,
            modality="multimodal",
        )
        result = extractor.extract(doc)
        assert isinstance(result, SimpleDocument)


@pytest.mark.integration
class TestAgentic:
    def test_text_returns_valid_schema(self, sample_pdf_path: Path) -> None:
        doc = PDFLoader().load(sample_pdf_path)
        extractor = AgenticExtractor(
            model=_make_model(),
            schema=SimpleDocument,
            modality="text",
            max_iterations=20,
        )
        result = extractor.extract(doc)
        assert isinstance(result, SimpleDocument)


@pytest.mark.integration
class TestEndToEnd:
    def test_loader_to_extractor_pipeline(self, sample_pdf_path: Path) -> None:
        doc = PDFLoader().load(sample_pdf_path)
        assert doc.page_count >= 1
        assert len(doc.full_text) > 0

        extractor = SinglePassExtractor(
            model=_make_model(),
            schema=SimpleDocument,
        )
        result = extractor.extract(doc)

        assert isinstance(result, SimpleDocument)
        # At least one field should be populated
        assert result.title is not None or len(result.parties) > 0
