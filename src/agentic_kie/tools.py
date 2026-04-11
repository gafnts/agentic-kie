"""
Document-scoped tools for the agentic extraction loop.

:func:`create_document_tools` is a factory that returns LangChain tool
closures bound to a single :class:`~agentic_kie.document.PDFDocument`.
The modality parameter controls which tools are exposed, so the agent
only sees the document representations it is configured to use.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import BaseTool, tool

from agentic_kie.document import PDFDocument


def create_document_tools(
    document: PDFDocument,
    *,
    modality: Literal["text", "image", "multimodal"] = "text",
) -> list[BaseTool]:
    """
    Factory for tools that wrap a single PDFDocument.

    Each tool is a closure over *document*, so the returned list is
    scoped to one extraction run.  The agent uses these tools to
    explore the document.

    Parameters
    ----------
    document:
        The PDF document the tools will operate on.
    modality:
        Controls which document tools are exposed. ``"text"`` includes
        only ``read_text``, ``"image"`` includes only ``load_images``,
        and ``"multimodal"`` includes both. ``get_page_count`` is
        always included. Defaults to ``"text"``.
    """

    @tool
    def get_page_count() -> int:
        """
        Return the total number of pages in the PDF document.
        """
        return document.page_count

    @tool
    def read_text(start: int, end: int | None = None) -> str:
        """
        Read text content from the PDF document. Pages are 0-indexed.
        If end is not provided, reads a single page.
        """
        try:
            return document.read_text(start, end)
        except ValueError as exc:
            return f"Error: {exc}"

    @tool
    def load_images(start: int, end: int | None = None) -> list[dict[str, object]]:
        """
        Get rendered page images from the PDF document. Pages are 0-indexed.
        If end is not provided, renders a single page.
        """
        try:
            images = document.load_images(start, end)
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
                for b64 in images
            ]
        except ValueError as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

    tools: list[BaseTool] = [get_page_count]

    if modality in ("text", "multimodal"):
        tools.append(read_text)
    if modality in ("image", "multimodal"):
        tools.append(load_images)

    return tools
