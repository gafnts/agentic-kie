"""
LangGraph Studio entrypoint.

Exposes a representative compiled agent graph for use with `uv run langgraph dev`.
Uses placeholder document tools that mirror the real tool signatures so the
graph structure renders correctly in the UI.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel

from agentic_kie.prompts import AGENTIC_SYSTEM_PROMPT


class _Placeholder(BaseModel):
    value: str


@tool
def get_page_count() -> int:
    """Return the total number of pages in the PDF document."""
    return 0


@tool
def read_text(start: int, end: int | None = None) -> str:
    """Read text content from the PDF document. Pages are 0-indexed.
    If end is not provided, reads a single page."""
    return ""


@tool
def load_images(start: int, end: int | None = None) -> list[str]:
    """Get base64-encoded PNG renders of pages. Pages are 0-indexed.
    If end is not provided, renders a single page."""
    return []


agent = create_agent(
    model="anthropic:claude-haiku-4-5",
    tools=[get_page_count, read_text, load_images],
    system_prompt=AGENTIC_SYSTEM_PROMPT,
    response_format=_Placeholder,
)
