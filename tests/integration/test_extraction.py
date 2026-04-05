from __future__ import annotations

import os
from pathlib import Path

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from agentic_kie import AgenticExtractor, PDFLoader, SinglePassExtractor

from .schema import SimpleDocument


def _make_model() -> BaseChatModel:
    """Build a simple chat model."""

    return ChatAnthropic(
        model="claude-haiku-4-5",
        api_key=os.environ["INTEGRATION_API_KEY"],  # type: ignore[call-arg]
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

    def test_image_returns_valid_schema(self, sample_pdf_path: Path) -> None:
        doc = PDFLoader().load(sample_pdf_path)
        extractor = SinglePassExtractor(
            model=_make_model(),
            schema=SimpleDocument,
            modality="image",
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

    def test_image_returns_valid_schema(self, sample_pdf_path: Path) -> None:
        doc = PDFLoader().load(sample_pdf_path)
        extractor = AgenticExtractor(
            model=_make_model(),
            schema=SimpleDocument,
            modality="image",
            max_iterations=20,
        )
        result = extractor.extract(doc)
        assert isinstance(result, SimpleDocument)
