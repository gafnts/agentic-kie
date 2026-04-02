"""
Exception hierarchy for document loading and extraction.

All document-level failures derive from :class:`DocumentLoadError`, making
them easy to catch together or individually. Extraction failures (e.g. an
agent exceeding its iteration limit) derive from :class:`ExtractionError`.
"""


class DocumentLoadError(Exception):
    """Base exception for all document loading failures."""


class CorruptDocumentError(DocumentLoadError):
    """PDF bytes could not be parsed."""


class PasswordProtectedError(DocumentLoadError):
    """PDF is encrypted and requires a password."""


class EmptyDocumentError(DocumentLoadError):
    """Document has zero pages, or no extractable text after OCR."""


class OCRNotConfiguredError(DocumentLoadError):
    """Scanned document detected but no OCR provider was supplied."""


class ExtractionError(Exception):
    """Base exception for extraction failures (e.g. agent exceeded max iterations)."""
