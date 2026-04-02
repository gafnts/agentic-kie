"""
Pluggable OCR provider protocol.

Defines the structural contract that any OCR backend must satisfy.
"""

from __future__ import annotations

from typing import Protocol


class OCRProvider(Protocol):
    """Contract for pluggable OCR backends (Textract, Tesseract, etc.)."""

    def extract_text(self, image: bytes) -> str: ...
