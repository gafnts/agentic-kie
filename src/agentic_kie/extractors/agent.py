"""Agentic extraction strategy using a LangGraph tool-calling loop."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, TypeVar, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
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

    Builds a LangGraph tool-calling loop that lets the model explore
    a PDF document page-by-page, then produces structured output
    against the target schema once the agent decides it has gathered
    enough information.

    The graph is compiled per-document because the tools are closures
    bound to a specific ``PDFDocument`` instance.  Compilation is a
    lightweight in-memory operation, negligible compared to LLM
    latency.

    Parameters
    ----------
    model:
        A LangChain chat model (ChatAnthropic, ChatOpenAI,
        ChatBedrock, ChatGoogleGenerativeAI, etc.).
    schema:
        The Pydantic model class defining extraction targets.
    modality:
        ``"text"`` exposes only text tools; ``"multimodal"`` also
        exposes a ``load_image`` tool for vision-capable models.
        Defaults to ``"text"``.
    system_prompt:
        Override the default agentic system prompt.
    max_iterations:
        Maximum number of graph steps (each agent turn consumes
        two steps: one for the model call, one for tool execution).
        Defaults to 50 (~25 agent turns).
    max_retries:
        Retry count for the final structured-output call.
        Defaults to 3.
    """

    def __init__(
        self,
        model: BaseChatModel,
        schema: type[T],
        *,
        modality: Literal["text", "multimodal"] = "text",
        system_prompt: str | None = None,
        max_iterations: int = 50,
        max_retries: int = 3,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self._model = model
        self._schema = schema
        self._modality = modality
        self._system_prompt = system_prompt or AGENTIC_SYSTEM_PROMPT
        self._max_iterations = max_iterations
        self._finalize_chain: Runnable[Any, Any] = model.with_structured_output(
            schema
        ).with_retry(stop_after_attempt=max_retries + 1)

    def extract(self, document: PDFDocument) -> T:
        """
        Extract structured data from a document using an agentic loop.

        The agent iteratively reads pages via tool calls, then a
        finalize step produces validated structured output.

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
            "Starting agentic extraction of %s from %d-page document (modality=%s)",
            self._schema.__name__,
            document.page_count,
            self._modality,
        )

        graph = self._build_graph(document)
        initial_message = self._build_initial_message()

        try:
            final_state: dict[str, Any] = graph.invoke(
                {"messages": [initial_message]},
                config={"recursion_limit": self._max_iterations},
            )
        except Exception as exc:
            if "recursion" in type(exc).__name__.lower():
                raise ExtractionError(
                    f"Agent exceeded {self._max_iterations} iterations "
                    f"without completing extraction of {self._schema.__name__}."
                ) from exc
            raise

        messages = [
            SystemMessage(content=self._system_prompt),
            *final_state["messages"],
        ]
        result = cast(T, self._finalize_chain.invoke(messages))

        logger.info("Agentic extraction complete for %s", self._schema.__name__)
        return result

    def _build_graph(self, document: PDFDocument) -> Any:
        """Compile a LangGraph StateGraph scoped to *document*."""
        tools = create_document_tools(
            document, include_images=(self._modality == "multimodal")
        )
        model_with_tools = self._model.bind_tools(tools)
        system_prompt = self._system_prompt

        def agent_node(state: MessagesState) -> dict[str, Any]:
            messages = [SystemMessage(content=system_prompt), *state["messages"]]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        graph = StateGraph(MessagesState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode(tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")
        return graph.compile()

    def _build_initial_message(self) -> HumanMessage:
        """Create the opening message that describes the extraction target."""
        fields = {
            name: info.description or str(info.annotation)
            for name, info in self._schema.model_fields.items()
        }
        return HumanMessage(
            content=(
                "Extract the following fields from this document:\n\n"
                f"{json.dumps(fields, indent=2)}"
            ),
        )
