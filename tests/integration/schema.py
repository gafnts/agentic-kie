from pydantic import BaseModel


class SimpleDocument(BaseModel):
    title: str | None = None
    date: str | None = None
    parties: list[str] = []
