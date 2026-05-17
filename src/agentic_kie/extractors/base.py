"""
Extractor protocol defining the contract for all extraction strategies.

The protocol enables type-safe dispatch without coupling strategies through
inheritance. Both :class:`~agentic_kie.extractors.single_pass.SinglePassExtractor`
and :class:`~agentic_kie.extractors.agent.AgenticExtractor` satisfy it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from langchain_core.messages.ai import UsageMetadata, add_usage
from pydantic import BaseModel

from agentic_kie.document import PDFDocument


@dataclass(frozen=True)
class ExtractionResult[T: BaseModel]:
    """
    Output of an extraction strategy.

    Carries the validated schema instance alongside token-usage metadata
    so callers (Lambdas, batch jobs, evaluation harnesses) can log cost
    and throughput without re-instrumenting the LLM call.

    Attributes
    ----------
    value:
        Validated instance of the target Pydantic schema.
    usage:
        Aggregated token usage across every LLM call made during the
        extraction. Mirrors LangChain's ``UsageMetadata`` shape:
        ``input_tokens``, ``output_tokens``, ``total_tokens``, and the
        optional ``input_token_details`` / ``output_token_details``
        breakdowns for cache and reasoning tokens.
    """

    value: T
    usage: UsageMetadata


def _fold_usage(per_model: dict[str, UsageMetadata]) -> UsageMetadata:
    """
    Collapse a per-model usage map into a single :class:`UsageMetadata`.

    ``get_usage_metadata_callback`` keys usage by model name; the
    extractors expose a single aggregate so callers don't have to know
    or care about model identity at the result boundary.
    """
    folded: UsageMetadata = UsageMetadata(
        input_tokens=0, output_tokens=0, total_tokens=0
    )
    for usage in per_model.values():
        folded = add_usage(folded, usage)
    return folded


@runtime_checkable
class Extractor[T: BaseModel](Protocol):
    """
    Contract for all extraction strategies.

    Both single-pass and agentic extractors satisfy this protocol,
    enabling type-safe dispatch at the routing layer without
    coupling the strategies through inheritance.
    """

    def extract(self, document: PDFDocument) -> ExtractionResult[T]: ...
