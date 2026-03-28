from __future__ import annotations

from typing import Any, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from agentic_kie.document import PDFDocument
from agentic_kie.prompts import SINGLE_PASS_SYSTEM_PROMPT

T = TypeVar("T", bound=BaseModel)


class SinglePassExtractor[T: BaseModel]:
    """
    Single-pass extraction strategy.

    Builds a prompt from the full document content — text only, or text
    plus rendered page images when multimodal — invokes the model with
    structured output bound to the target schema, and returns a validated
    Pydantic instance.

    The structured-output chain is built once at construction time and
    reused across documents, avoiding redundant setup when processing
    large evaluation sets.

    Parameters
    ----------
    model:
        A LangChain chat model (ChatBedrock, ChatOpenAI,
        ChatAnthropic, ChatGoogleGenerativeAI, etc.).
    schema:
        The Pydantic model class defining extraction targets.
        Field names, types, and descriptions are forwarded to
        the model via LangChain's structured output binding.
    multimodal:
        If True, include all page images alongside the text
        in the prompt. Defaults to text-only.
    system_prompt:
        Override the default system prompt. If None, uses
        the built-in extraction prompt.
    """

    def __init__(
        self,
        model: BaseChatModel,
        schema: type[T],
        *,
        multimodal: bool = False,
        system_prompt: str | None = None,
    ) -> None:
        self._schema = schema
        self._multimodal = multimodal
        self._system_prompt = system_prompt or SINGLE_PASS_SYSTEM_PROMPT
        self._chain: Runnable[Any, Any] = model.with_structured_output(schema)

    def extract(self, document: PDFDocument) -> T:
        """
        Extract structured data from a document in a single LLM call.

        Parameters
        ----------
        document:
            A validated PDFDocument to extract from.

        Returns
        -------
            A validated instance of the target schema.
        """
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=self._build_content(document)),
        ]

        return cast(T, self._chain.invoke(messages))

    def _build_content(self, document: PDFDocument) -> str | list[str | dict[Any, Any]]:
        """
        Build the content payload for the human message.

        For text-only mode, returns the full document text as a string.
        For multimodal mode, returns a list of content blocks: the full
        text first, followed by one base64-encoded PNG per page.
        """
        if not self._multimodal:
            return document.full_text

        content: list[str | dict[Any, Any]] = [
            {"type": "text", "text": document.full_text},
        ]

        for b64 in document.all_images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        return content
