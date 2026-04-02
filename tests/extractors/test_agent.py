from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents.middleware import ModelRetryMiddleware
from pydantic import BaseModel

from agentic_kie.document import PDFDocument
from agentic_kie.exceptions import ExtractionError
from agentic_kie.extractors.agent import AgenticExtractor
from agentic_kie.prompts import AGENTIC_SYSTEM_PROMPT


class _Schema(BaseModel):
    name: str | None = None
    value: int | None = None


_EXPECTED = _Schema(name="test", value=42)


@pytest.fixture
def mock_model() -> MagicMock:
    return MagicMock()


class TestInit:
    def test_uses_default_system_prompt(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        assert extractor._system_prompt == AGENTIC_SYSTEM_PROMPT

    def test_accepts_custom_system_prompt(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(
            model=mock_model, schema=_Schema, system_prompt="Custom"
        )
        assert extractor._system_prompt == "Custom"

    def test_defaults_to_text_modality(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        assert extractor._modality == "text"

    def test_accepts_modality(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        assert extractor._modality == "multimodal"

    def test_zero_max_iterations_raises(self, mock_model: MagicMock) -> None:
        with pytest.raises(ValueError):
            AgenticExtractor(model=mock_model, schema=_Schema, max_iterations=0)

    def test_negative_max_retries_raises(self, mock_model: MagicMock) -> None:
        with pytest.raises(ValueError):
            AgenticExtractor(model=mock_model, schema=_Schema, max_retries=-1)

    def test_default_max_iterations(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        assert extractor._max_iterations == 50

    def test_default_max_retries(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        assert extractor._max_retries == 3

    def test_stores_schema(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        assert extractor._schema is _Schema


class TestExtract:
    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_returns_schema_instance(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        result = extractor.extract(pdf_document)

        assert isinstance(result, _Schema)
        assert result.name == "test"
        assert result.value == 42

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_invokes_agent_once(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        extractor.extract(pdf_document)

        mock_agent.invoke.assert_called_once()

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_passes_schema_as_response_format(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        extractor.extract(pdf_document)

        mock_create_agent.assert_called_once()
        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["model"] is mock_model
        assert call_kwargs["tools"] == []
        assert call_kwargs["system_prompt"] == AGENTIC_SYSTEM_PROMPT
        assert call_kwargs["response_format"] is _Schema
        assert len(call_kwargs["middleware"]) == 1
        assert isinstance(call_kwargs["middleware"][0], ModelRetryMiddleware)

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_zero_retries_passes_empty_middleware(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema, max_retries=0)
        extractor.extract(pdf_document)

        call_kwargs = mock_create_agent.call_args[1]
        assert call_kwargs["middleware"] == []

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_passes_recursion_limit(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(
            model=mock_model, schema=_Schema, max_iterations=30
        )
        extractor.extract(pdf_document)

        config = mock_agent.invoke.call_args[1]["config"]
        assert config["recursion_limit"] == 30

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_multimodal_passes_modality(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        extractor.extract(pdf_document)

        mock_create_tools.assert_called_once_with(pdf_document, modality="multimodal")

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_image_mode_passes_modality(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema, modality="image")
        extractor.extract(pdf_document)

        mock_create_tools.assert_called_once_with(pdf_document, modality="image")

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_text_mode_excludes_images(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        extractor.extract(pdf_document)

        mock_create_tools.assert_called_once_with(pdf_document, modality="text")

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_recursion_error_raises_extraction_error(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []

        class GraphRecursionError(Exception):
            pass

        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = GraphRecursionError("limit reached")
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with pytest.raises(ExtractionError, match="exceeded"):
            extractor.extract(pdf_document)

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_non_recursion_error_propagates(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("connection failed")
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with pytest.raises(RuntimeError, match="connection failed"):
            extractor.extract(pdf_document)


class TestLogging:
    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_logs_extraction_start_at_info(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with caplog.at_level(logging.INFO):
            extractor.extract(pdf_document)

        assert any(
            "_Schema" in r.message and "3-page" in r.message for r in caplog.records
        )

    @patch("agentic_kie.extractors.agent.create_agent")
    @patch("agentic_kie.extractors.agent.create_document_tools")
    def test_logs_extraction_complete_at_info(
        self,
        mock_create_tools: MagicMock,
        mock_create_agent: MagicMock,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_create_tools.return_value = []
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {"structured_response": _EXPECTED}
        mock_create_agent.return_value = mock_agent

        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with caplog.at_level(logging.INFO):
            extractor.extract(pdf_document)

        assert any(
            "complete" in r.message.lower() and "_Schema" in r.message
            for r in caplog.records
        )
