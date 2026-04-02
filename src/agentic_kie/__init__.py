"""
Agentic KIE — structured key information extraction from PDF documents.

Provides a two-strategy extraction pipeline built on LangChain:

- :class:`SinglePassExtractor` issues a single structured LLM call.
- :class:`AgenticExtractor` runs a ReAct agent loop with document tools.

Both satisfy the :class:`Extractor` protocol and return validated Pydantic
instances, so callers can swap strategies without changing downstream code.

Document ingestion is handled by :class:`PDFLoader`, which detects native
text layers, routes to a pluggable :class:`OCRProvider` when needed, and
returns an immutable :class:`PDFDocument`.
"""

from importlib.metadata import version

__version__ = version("agentic-kie")


from .document import PDFDocument
from .exceptions import (
    CorruptDocumentError,
    DocumentLoadError,
    EmptyDocumentError,
    ExtractionError,
    OCRNotConfiguredError,
    PasswordProtectedError,
)
from .extractors.agent import AgenticExtractor
from .extractors.base import Extractor
from .extractors.single_pass import SinglePassExtractor
from .loader import PDFLoader
from .ocr import OCRProvider
from .tools import create_document_tools

__all__ = [
    "PDFLoader",
    "PDFDocument",
    "Extractor",
    "SinglePassExtractor",
    "AgenticExtractor",
    "create_document_tools",
    "OCRProvider",
    "DocumentLoadError",
    "CorruptDocumentError",
    "EmptyDocumentError",
    "OCRNotConfiguredError",
    "PasswordProtectedError",
    "ExtractionError",
]
