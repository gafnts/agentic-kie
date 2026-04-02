from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from agentic_kie.extractors.agent import AgenticExtractor
from agentic_kie.extractors.base import Extractor
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
