from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from agentic_kie.document import PDFDocument
from agentic_kie.exceptions import ExtractionError
from agentic_kie.extractors.agent import AgenticExtractor
from agentic_kie.extractors.base import Extractor
from agentic_kie.prompts import AGENTIC_SYSTEM_PROMPT


class _Schema(BaseModel):
    name: str | None = None
    value: int | None = None


_EXPECTED = _Schema(name="test", value=42)


@pytest.fixture
def mock_finalize_chain() -> MagicMock:
    chain = MagicMock()
    chain.invoke.return_value = _EXPECTED
    return chain


@pytest.fixture
def mock_model(mock_finalize_chain: MagicMock) -> MagicMock:
    model = MagicMock()
    structured = MagicMock()
    structured.with_retry.return_value = mock_finalize_chain
    model.with_structured_output.return_value = structured
    model.bind_tools.return_value = model
    model.invoke.return_value = MagicMock(tool_calls=[])
    return model


class TestInit:
    def test_binds_schema_for_finalize(self, mock_model: MagicMock) -> None:
        AgenticExtractor(model=mock_model, schema=_Schema)
        mock_model.with_structured_output.assert_called_once_with(_Schema)

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

    def test_negative_max_iterations_raises(self, mock_model: MagicMock) -> None:
        with pytest.raises(ValueError):
            AgenticExtractor(model=mock_model, schema=_Schema, max_iterations=0)

    def test_negative_max_retries_raises(self, mock_model: MagicMock) -> None:
        with pytest.raises(ValueError):
            AgenticExtractor(model=mock_model, schema=_Schema, max_retries=-1)

    def test_default_max_retries_composes_retry(self, mock_model: MagicMock) -> None:
        AgenticExtractor(model=mock_model, schema=_Schema)
        structured = mock_model.with_structured_output.return_value
        structured.with_retry.assert_called_once_with(stop_after_attempt=4)


class TestProtocol:
    def test_satisfies_extractor_protocol(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        assert isinstance(extractor, Extractor)


class TestBuildGraph:
    def test_returns_compiled_graph(
        self, mock_model: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        graph = extractor._build_graph(pdf_document)
        assert hasattr(graph, "invoke")

    def test_binds_tools_to_model(
        self, mock_model: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        extractor._build_graph(pdf_document)
        mock_model.bind_tools.assert_called_once()
        tools = mock_model.bind_tools.call_args[0][0]
        names = {t.name for t in tools}
        assert "get_page_count" in names
        assert "read_text" in names

    def test_text_mode_excludes_image_tool(
        self, mock_model: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema, modality="text")
        extractor._build_graph(pdf_document)
        tools = mock_model.bind_tools.call_args[0][0]
        names = {t.name for t in tools}
        assert "load_image" not in names

    def test_multimodal_includes_image_tool(
        self, mock_model: MagicMock, pdf_document: PDFDocument
    ) -> None:
        extractor = AgenticExtractor(
            model=mock_model, schema=_Schema, modality="multimodal"
        )
        extractor._build_graph(pdf_document)
        tools = mock_model.bind_tools.call_args[0][0]
        names = {t.name for t in tools}
        assert "load_images" in names


class TestBuildInitialMessage:
    def test_mentions_schema_fields(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        msg = extractor._build_initial_message()
        assert "name" in msg.content
        assert "value" in msg.content

    def test_instructs_extraction(self, mock_model: MagicMock) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        msg = extractor._build_initial_message()
        assert "Extract" in msg.content


class TestExtract:
    def test_returns_schema_instance(
        self,
        mock_model: MagicMock,
        mock_finalize_chain: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with patch.object(extractor, "_build_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"messages": []}
            mock_build.return_value = mock_graph
            result = extractor.extract(pdf_document)

        assert isinstance(result, _Schema)
        assert result.name == "test"
        assert result.value == 42

    def test_invokes_graph_with_recursion_limit(
        self,
        mock_model: MagicMock,
        mock_finalize_chain: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        extractor = AgenticExtractor(
            model=mock_model, schema=_Schema, max_iterations=30
        )
        with patch.object(extractor, "_build_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"messages": []}
            mock_build.return_value = mock_graph
            extractor.extract(pdf_document)

        config = mock_graph.invoke.call_args[1]["config"]
        assert config["recursion_limit"] == 30

    def test_invokes_finalize_chain(
        self,
        mock_model: MagicMock,
        mock_finalize_chain: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with patch.object(extractor, "_build_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"messages": []}
            mock_build.return_value = mock_graph
            extractor.extract(pdf_document)

        mock_finalize_chain.invoke.assert_called_once()

    def test_recursion_error_raises_extraction_error(
        self,
        mock_model: MagicMock,
        pdf_document: PDFDocument,
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)

        class GraphRecursionError(Exception):
            pass

        with patch.object(extractor, "_build_graph") as mock_build:
            mock_graph = MagicMock()
            mock_graph.invoke.side_effect = GraphRecursionError("limit reached")
            mock_build.return_value = mock_graph

            with pytest.raises(ExtractionError, match="exceeded"):
                extractor.extract(pdf_document)


class TestLogging:
    def test_logs_extraction_start(
        self,
        mock_model: MagicMock,
        mock_finalize_chain: MagicMock,
        pdf_document: PDFDocument,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with (
            patch.object(extractor, "_build_graph") as mock_build,
            caplog.at_level(logging.INFO),
        ):
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"messages": []}
            mock_build.return_value = mock_graph
            extractor.extract(pdf_document)

        assert any(
            "_Schema" in r.message and "3-page" in r.message for r in caplog.records
        )

    def test_logs_extraction_complete(
        self,
        mock_model: MagicMock,
        mock_finalize_chain: MagicMock,
        pdf_document: PDFDocument,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        extractor = AgenticExtractor(model=mock_model, schema=_Schema)
        with (
            patch.object(extractor, "_build_graph") as mock_build,
            caplog.at_level(logging.INFO),
        ):
            mock_graph = MagicMock()
            mock_graph.invoke.return_value = {"messages": []}
            mock_build.return_value = mock_graph
            extractor.extract(pdf_document)

        assert any(
            "complete" in r.message.lower() and "_Schema" in r.message
            for r in caplog.records
        )
