"""Document-scoped tools for the agentic extraction loop."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agentic_kie.document import PDFDocument


def create_document_tools(
    document: PDFDocument,
    *,
    include_images: bool = False,
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
    include_images:
        When True, a ``load_images`` tool is added for vision-capable
        models.  Defaults to False.
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
    def load_images(start: int, end: int | None = None) -> list[str]:
        """
        Get base64-encoded PNG renders of pages. Pages are 0-indexed.
        If end is not provided, renders a single page.
        """
        try:
            return document.load_images(start, end)
        except ValueError as exc:
            return [f"Error: {exc}"]

    tools: list[BaseTool] = [get_page_count, read_text]

    if include_images:
        tools.append(load_images)

    return tools
