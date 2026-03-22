from importlib.metadata import version

__version__ = version("agentic-kie")


from .reader import PDFReader

__all__ = ["PDFReader"]
