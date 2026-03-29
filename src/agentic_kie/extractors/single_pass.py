from __future__ import annotations

from typing import Any, Literal, TypeVar, cast

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

    Builds a prompt from the full document content — text only, page
    images only, or both — invokes the model with structured output
    bound to the target schema, and returns a validated Pydantic
    instance.

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
    modality:
        Controls which document representations are sent to the
        model. ``"text"`` sends only the extracted text,
        ``"image"`` sends only rendered page images, and
        ``"multimodal"`` sends the text followed by the images.
        Defaults to ``"text"``.
    system_prompt:
        Override the default system prompt. If None, uses
        the built-in extraction prompt.
    """

    def __init__(
        self,
        model: BaseChatModel,
        schema: type[T],
        *,
        modality: Literal["text", "image", "multimodal"] = "text",
        system_prompt: str | None = None,
    ) -> None:
        self._schema = schema
        self._modality = modality
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

    def _build_content(self, document: PDFDocument) -> str | list[str | dict[str, Any]]:
        """
        Build the content payload for the human message.

        For ``"text"`` modality, returns the full document text as a string.
        For ``"image"``, returns a list with one base64-encoded PNG block
        per page. For ``"multimodal"``, returns the full text block followed
        by the image blocks.
        """
        if self._modality == "text":
            return document.full_text

        image_blocks: list[str | dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
            for b64 in document.all_images
        ]

        if self._modality == "image":
            return image_blocks

        return [{"type": "text", "text": document.full_text}, *image_blocks]
