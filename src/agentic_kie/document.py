"""
Immutable document representation exposing text and vision modalities.

:class:`PDFDocument` is the agent-facing surface of the library. It holds
per-page text and raw PDF bytes, and renders pages to base64 PNG on demand.
"""

from __future__ import annotations

import base64
import functools

import pymupdf


class PDFDocument:
    """
    Immutable document representation exposing text and vision modalities.

    Holds per-page text and raw PDF bytes in memory so that any page range
    can be read or rendered as many times as needed without additional I/O.
    Images are rendered lazily and cached on first access.

    Parameters
    ----------
    pdf_text:
        Per-page text content, one string per page.
    pdf_bytes:
        Raw PDF bytes used for on-demand image rendering.
    dpi:
        Resolution for rendering pages to images.
    ocr:
        Whether the text was produced by OCR rather than a native text layer.
    """

    def __init__(
        self,
        pdf_text: list[str],
        pdf_bytes: bytes,
        *,
        dpi: int = 150,
        ocr: bool = False,
    ) -> None:
        self._pdf_text = pdf_text
        self._pdf_bytes = pdf_bytes
        self._dpi = dpi
        self._ocr = ocr

    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return len(self._pdf_text)

    @property
    def is_ocr(self) -> bool:
        """True if text was extracted via OCR rather than a native text layer."""
        return self._ocr

    @property
    def full_text(self) -> str:
        """All page text concatenated with double newlines."""
        return "\n\n".join(self._pdf_text)

    @functools.cached_property
    def all_images(self) -> list[str]:
        """All pages rendered as base64-encoded PNG strings, cached on first access."""
        if self.page_count == 0:
            return []
        return self.load_images(0, self.page_count)

    def read_text(self, start: int, end: int | None = None) -> str:
        """
        Return the text of a page range as a single string.

        Parameters
        ----------
        start:
            Index of the first page (inclusive).
        end:
            Index of the last page (exclusive). Defaults to start + 1.

        Raises
        ------
        ValueError
            If the range is negative, empty, inverted, or out of bounds.
        """
        end = end if end is not None else start + 1
        self._validate_range(start, end)
        return "\n\n".join(self._pdf_text[start:end])

    def load_images(self, start: int, end: int | None = None) -> list[str]:
        """
        Render a page range as base64-encoded PNG strings.

        Parameters
        ----------
        start:
            Index of the first page (inclusive).
        end:
            Index of the last page (exclusive). Defaults to start + 1.

        Raises
        ------
        ValueError
            If the range is negative, empty, inverted, or out of bounds.
        """
        end = end if end is not None else start + 1
        self._validate_range(start, end)
        with pymupdf.open(stream=self._pdf_bytes, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
            return [self._page_to_png(doc[i], self._dpi) for i in range(start, end)]

    def _validate_range(self, start: int, end: int) -> None:
        """
        Assert that a half-open page range is valid for this document.

        Raises
        ------
        ValueError
            If either bound is negative, start equals or exceeds end, or the
            range extends beyond the document's page count.
        """
        if start < 0 or end < 0:
            raise ValueError(
                f"Negative indices are not supported (start={start}, end={end})."
            )
        if start >= end:
            raise ValueError(f"start ({start}) must be less than end ({end}).")
        if start >= self.page_count or end > self.page_count:
            raise ValueError(
                f"Range ({start}, {end}) is out of bounds for a {self.page_count}-page document."
            )

    @staticmethod
    def _page_to_png(page: pymupdf.Page, dpi: int) -> str:
        """Render a single page to a base64-encoded PNG string at the given DPI."""
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)  # type: ignore[no-untyped-call]
        pixmap = page.get_pixmap(matrix=matrix)
        return base64.b64encode(pixmap.tobytes("png")).decode()  # type: ignore[no-untyped-call]
