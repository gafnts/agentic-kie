from importlib.metadata import version

__version__ = version("agentic-kie")


from .document import PDFDocument
from .exceptions import (
    CorruptDocumentError,
    DocumentLoadError,
    EmptyDocumentError,
    OCRNotConfiguredError,
    PasswordProtectedError,
)
from .extractors.base import Extractor
from .extractors.single_pass import SinglePassExtractor
from .loader import PDFLoader
from .ocr import OCRProvider

__all__ = [
    "PDFDocument",
    "CorruptDocumentError",
    "DocumentLoadError",
    "EmptyDocumentError",
    "OCRNotConfiguredError",
    "PasswordProtectedError",
    "Extractor",
    "SinglePassExtractor",
    "PDFLoader",
    "OCRProvider",
]
