from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages.ai import UsageMetadata
from pydantic import BaseModel

from agentic_kie.extractors.agent import AgenticExtractor
from agentic_kie.extractors.base import Extractor, _fold_usage
from agentic_kie.extractors.single_pass import SinglePassExtractor


class _Schema(BaseModel):
    name: str | None = None


class TestExtractorProtocol:
    def test_single_pass_satisfies_protocol(self) -> None:
        model = MagicMock()
        model.with_structured_output.return_value = MagicMock()
        extractor = SinglePassExtractor(model=model, schema=_Schema)
        assert isinstance(extractor, Extractor)

    def test_agentic_satisfies_protocol(self) -> None:
        model = MagicMock()
        extractor = AgenticExtractor(model=model, schema=_Schema)
        assert isinstance(extractor, Extractor)


class TestFoldUsage:
    def test_empty_map_returns_zero_usage(self) -> None:
        folded = _fold_usage({})
        assert folded["input_tokens"] == 0
        assert folded["output_tokens"] == 0
        assert folded["total_tokens"] == 0

    def test_single_model_is_returned_as_is(self) -> None:
        usage = UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30)
        folded = _fold_usage({"gpt-4": usage})
        assert folded["input_tokens"] == 10
        assert folded["output_tokens"] == 20
        assert folded["total_tokens"] == 30

    def test_multiple_models_are_summed(self) -> None:
        per_model = {
            "gpt-4": UsageMetadata(input_tokens=10, output_tokens=20, total_tokens=30),
            "claude": UsageMetadata(input_tokens=5, output_tokens=7, total_tokens=12),
        }
        folded = _fold_usage(per_model)
        assert folded["input_tokens"] == 15
        assert folded["output_tokens"] == 27
        assert folded["total_tokens"] == 42
