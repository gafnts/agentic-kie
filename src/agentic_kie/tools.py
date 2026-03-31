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
    Build LangChain tools that wrap a single PDFDocument.

    Each tool is a closure over *document*, so the returned list is
    scoped to one extraction run.  The agent uses these tools to
    explore the document page-by-page.

    Parameters
    ----------
    document:
        The document the tools will operate on.
    include_images:
        When True, a ``load_image`` tool is added for vision-capable
        models.  Defaults to False.
    """

    @tool
    def get_page_count() -> int:
        """Return the total number of pages in the document."""
        return document.page_count

    @tool
    def read_text(page_number: int) -> str:
        """Return the text content of a single page.

        Pages are zero-indexed (0 is the first page).
        """
        try:
            return document.read_text(page_number)
        except ValueError as exc:
            return f"Error: {exc}"

    @tool
    def load_image(page_number: int) -> list[dict[str, str | dict[str, str]]]:
        """Render a single page as an image for visual inspection.

        Pages are zero-indexed (0 is the first page).
        Returns image content blocks that the model can see.
        """
        try:
            images = document.load_images(page_number)
            return [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
                for b64 in images
            ]
        except ValueError as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

    tools: list[BaseTool] = [get_page_count, read_text]
    if include_images:
        tools.append(load_image)
    return tools
