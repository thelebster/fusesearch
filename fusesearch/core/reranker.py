import os
from abc import ABC, abstractmethod


class Reranker(ABC):
    """Abstract base class for reranking providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'local', 'cohere')."""

    @abstractmethod
    def rerank(
        self, query: str, results: list[dict], limit: int | None = None
    ) -> list[dict]:
        """Rerank search results by relevance to query.

        Args:
            query: The search query.
            results: List of search result dicts (must have 'content' key).
            limit: Max results to return. None = return all, reordered.

        Returns:
            Results reordered by relevance, with 'score' updated.
        """


class LocalReranker(Reranker):
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(
        self, model: str | None = None, local_files_only: bool = False
    ):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "Local reranker requires the [local] extra. "
                "Install with: pip install fusesearch[local]"
            ) from None

        model = model or os.getenv(
            "FUSESEARCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.model = CrossEncoder(model, local_files_only=local_files_only)

    @property
    def name(self) -> str:
        return "local"

    def rerank(
        self, query: str, results: list[dict], limit: int | None = None
    ) -> list[dict]:
        if not results:
            return results

        pairs = [(query, r["content"]) for r in results]
        scores = self.model.predict(pairs)

        scored = [
            {**result, "score": float(score)}
            for result, score in zip(results, scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        if limit is not None:
            scored = scored[:limit]
        return scored


def rerank_enabled() -> bool:
    """Check if reranking is enabled globally via FUSESEARCH_RERANK env var."""
    return os.getenv("FUSESEARCH_RERANK", "false").lower() == "true"


def create_reranker(provider: str | None = None) -> Reranker:
    """Create a reranker based on provider name or FUSESEARCH_RERANKER env var."""
    provider = provider or os.getenv("FUSESEARCH_RERANKER", "local")
    if provider == "local":
        return LocalReranker()
    raise ValueError(f"Unknown reranker provider: {provider}")
