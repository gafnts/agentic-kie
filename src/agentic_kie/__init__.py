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

__all__ = [
    "PDFLoader",
    "PDFDocument",
    "Extractor",
    "SinglePassExtractor",
    "AgenticExtractor",
    "OCRProvider",
    "DocumentLoadError",
    "CorruptDocumentError",
    "EmptyDocumentError",
    "OCRNotConfiguredError",
    "PasswordProtectedError",
    "ExtractionError",
]
