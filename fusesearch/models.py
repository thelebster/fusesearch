from datetime import datetime
from hashlib import sha256

from pydantic import BaseModel, Field, computed_field


class Document(BaseModel):
    """A normalized document from any source."""

    source_type: str
    source_id: str
    title: str
    content: str
    url: str | None = None
    metadata: dict = Field(default_factory=dict)
    fetched_at: datetime = Field(default_factory=datetime.now)


class Chunk(BaseModel):
    """A piece of a document, ready for embedding and indexing."""

    document_source_id: str
    source_type: str
    title: str
    content: str
    url: str | None = None
    metadata: dict = Field(default_factory=dict)
    heading_path: list[str] = Field(default_factory=list)
    chunk_index: int = 0

    @computed_field
    @property
    def content_hash(self) -> str:
        return sha256(self.content.encode()).hexdigest()
