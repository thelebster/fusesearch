import os
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used for collection naming (e.g. 'local', 'openai')."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class OpenAIEmbedder(Embedder):
    """OpenAI embedding provider."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI embedder requires the [openai] extra. "
                "Install with: pip install fusesearch[openai]"
            ) from None

        self.model = model
        self.client = OpenAI(api_key=api_key, max_retries=10)
        self._dimension = 1536

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class LocalEmbedder(Embedder):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model: str | None = None, local_files_only: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Local embedder requires the [local] extra. "
                "Install with: pip install fusesearch[local]"
            ) from None

        model = model or os.getenv("FUSESEARCH_EMBED_MODEL", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model, local_files_only=local_files_only)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def name(self) -> str:
        return "local"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()


class OllamaEmbedder(Embedder):
    """Ollama embedding provider."""

    def __init__(self, model: str | None = None, host: str | None = None):
        try:
            from ollama import Client
        except ImportError:
            raise ImportError(
                "Ollama embedder requires the [ollama] extra. "
                "Install with: pip install fusesearch[ollama]"
            ) from None

        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.client = Client(host=host or os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        # Probe to determine embedding dimension (varies by model)
        probe = self.client.embed(model=self.model, input=["dimension probe"])
        self._dimension = len(probe.embeddings[0])

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embed(model=self.model, input=texts)
        return [list(e) for e in response.embeddings]


def create_embedder(provider: str | None = None) -> Embedder:
    """Create an embedder based on provider name or FUSESEARCH_EMBEDDER env var."""
    provider = provider or os.getenv("FUSESEARCH_EMBEDDER", "local")
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "local":
        return LocalEmbedder()
    if provider == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"Unknown embedder provider: {provider}")
