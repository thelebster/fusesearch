import os
from abc import ABC, abstractmethod

class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class LocalEmbedder(Embedder):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model: str | None = None, local_files_only: bool = False):
        from sentence_transformers import SentenceTransformer

        model = model or os.getenv("FUSESEARCH_EMBED_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model, local_files_only=local_files_only)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
