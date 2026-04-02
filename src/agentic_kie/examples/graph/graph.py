"""
LangGraph Studio entrypoint.

Exposes a compiled agent graph for use with ``uv run langgraph dev``.
Loads a real PDF document and wires up Claude Haiku with the document
tools so the graph is fully functional in the UI.
"""

from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from nda import NDA

from agentic_kie.loader import PDFLoader
from agentic_kie.prompts import AGENTIC_SYSTEM_PROMPT
from agentic_kie.tools import create_document_tools

file_path = (
    Path().cwd()
    / "notebooks"
    / "data"
    / "train"
    / "documents"
    / "00a1d238e37ac225b8045a97953e845d.pdf"
)

loader = PDFLoader()
doc = loader.load(file_path)
tools = create_document_tools(doc, modality="text")

agent = create_agent(
    model=ChatAnthropic(model="claude-haiku-4-5"),
    tools=tools,
    system_prompt=AGENTIC_SYSTEM_PROMPT,
    response_format=NDA,
)
