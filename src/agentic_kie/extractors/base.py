from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from agentic_kie.document import PDFDocument

T = TypeVar("T", bound=BaseModel, covariant=True)


@runtime_checkable
class Extractor(Protocol[T]):
    """
    Contract for all extraction strategies.

    Both single-pass and agentic extractors satisfy this protocol,
    enabling type-safe dispatch at the routing layer without
    coupling the strategies through inheritance.
    """

    def extract(self, document: PDFDocument) -> T: ...
