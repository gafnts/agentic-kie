from importlib.metadata import version

__version__ = version("agentic-kie")


from .reader import PDFDocument

__all__ = ["PDFDocument"]
