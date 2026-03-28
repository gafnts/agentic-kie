from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

from agentic_kie.document import PDFDocument

T = TypeVar("T", bound=BaseModel, covariant=True)


class Extractor(Protocol[T]):
    def extract(self, document: PDFDocument) -> T: ...
