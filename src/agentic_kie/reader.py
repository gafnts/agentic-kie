from __future__ import annotations

import base64

import pymupdf


class PDFDocument:
    """Parsed PDF representation exposing text and vision modalities."""

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
        return len(self._pdf_text)

    @property
    def is_ocr(self) -> bool:
        return self._ocr

    @property
    def full_text(self) -> str:
        return "\n\n".join(self._pdf_text)

    @property
    def all_images(self) -> list[str]:
        return self.load_images(0, self.page_count)

    def read_text(self, start: int, end: int | None = None) -> str:
        end = end or start + 1
        return "\n\n".join(self._pdf_text[start:end])

    def load_images(self, start: int, end: int | None = None) -> list[str]:
        doc = pymupdf.open(stream=self._pdf_bytes, filetype="pdf")  # type: ignore[no-untyped-call]
        end = end or start + 1
        return [self._page_to_png(doc[i], self._dpi) for i in range(start, end)]

    @staticmethod
    def _page_to_png(page: pymupdf.Page, dpi: int) -> str:
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)  # type: ignore[no-untyped-call]
        pixmap = page.get_pixmap(matrix=matrix)
        return base64.b64encode(pixmap.tobytes("png")).decode()  # type: ignore[no-untyped-call]
