"""Agentic extraction strategy using LangChain's create_agent."""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from agentic_kie.document import PDFDocument
from agentic_kie.exceptions import ExtractionError
from agentic_kie.prompts import AGENTIC_SYSTEM_PROMPT
from agentic_kie.tools import create_document_tools

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class AgenticExtractor[T: BaseModel]:
    """
    Agentic extraction strategy.

    Uses ``create_agent`` from LangChain to build a ReAct agent that
    explores a PDF document via tools, then produces structured output
    against the target schema.

    Parameters
    ----------
    model:
        A LangChain chat model (ChatAnthropic, ChatOpenAI,
        ChatBedrock, ChatGoogleGenerativeAI, etc.).
    schema:
        The Pydantic model class defining extraction targets.
    multimodal:
        When ``True``, exposes a ``load_images`` tool for
        vision-capable models. Defaults to ``False``.
    system_prompt:
        Override the default agentic system prompt.
    max_iterations:
        Maximum number of agent steps before raising
        ``ExtractionError``. Defaults to 50.
    max_retries:
        Maximum number of retry attempts on transient model
        failures (rate limits, timeouts, overloaded errors).
        Defaults to 3.
    """

    def __init__(
        self,
        model: BaseChatModel,
        schema: type[T],
        *,
        multimodal: bool = False,
        system_prompt: str = AGENTIC_SYSTEM_PROMPT,
        max_iterations: int = 50,
        max_retries: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self._model = model
        self._schema = schema
        self._multimodal = multimodal
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._max_retries = max_retries

    def extract(self, document: PDFDocument) -> T:
        """
        Extract structured data from a document using an agentic loop.

        Parameters
        ----------
        document:
            A validated PDFDocument to extract from.

        Returns
        -------
            A validated instance of the target schema.

        Raises
        ------
        ExtractionError
            If the agent exceeds ``max_iterations`` without finishing.
        """
        logger.info(
            "Starting agentic extraction of %s from %d-page document",
            self._schema.__name__,
            document.page_count,
        )

        tools = create_document_tools(document, include_images=self._multimodal)

        middleware = []
        if self._max_retries > 0:
            middleware.append(ModelRetryMiddleware(max_retries=self._max_retries))

        agent = create_agent(
            model=self._model,
            tools=tools,
            system_prompt=self._system_prompt,
            response_format=self._schema,
            middleware=middleware,
        )

        try:
            result: dict[str, Any] = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Extract the target entities from this document."
                            ),
                        }
                    ]
                },
                config={"recursion_limit": self._max_iterations},
            )
        except Exception as exc:
            if "recursion" in type(exc).__name__.lower():
                raise ExtractionError(
                    f"Agent exceeded {self._max_iterations} iterations "
                    f"without completing extraction of {self._schema.__name__}."
                ) from exc
            raise

        structured = cast(T, result["structured_response"])
        logger.info("Agentic extraction complete for %s", self._schema.__name__)
        return structured
