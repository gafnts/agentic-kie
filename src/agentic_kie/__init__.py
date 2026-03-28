from importlib.metadata import version

__version__ = version("agentic-kie")


from .document import PDFDocument

__all__ = ["PDFDocument"]
