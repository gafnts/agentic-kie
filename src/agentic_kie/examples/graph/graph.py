"""
LangGraph Studio entrypoint.

Exposes a compiled agent graph for use with ``uv run langgraph dev``.
Loads a real PDF document and wires up Claude Haiku with the document
tools so the graph is fully functional in the UI.
"""

from __future__ import annotations

from pathlib import Path

from langchain_anthropic import ChatAnthropic
from nda import NDA

from agentic_kie.extractors.agent import AgenticExtractor
from agentic_kie.loader import PDFLoader

file_path = (
    Path().cwd()
    / "notebooks"
    / "data"
    / "train"
    / "documents"
    / "00a1d238e37ac225b8045a97953e845d.pdf"
)

doc = PDFLoader().load(file_path)

extractor = AgenticExtractor(model=ChatAnthropic(model="claude-haiku-4-5"), schema=NDA)

agent = extractor.build_graph(doc)
