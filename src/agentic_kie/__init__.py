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

__all__ = [
    "CorruptDocumentError",
    "DocumentLoadError",
    "EmptyDocumentError",
    "OCRNotConfiguredError",
    "PasswordProtectedError",
    "PDFDocument",
]
