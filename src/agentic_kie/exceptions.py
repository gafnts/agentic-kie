"""Domain-level exceptions for document loading and processing."""


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
