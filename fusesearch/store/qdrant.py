import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from fusesearch.models import Chunk


def hash_to_uuid(content_hash: str) -> str:
    """Convert a SHA-256 hex string to a UUID (uses first 32 hex chars)."""
    return str(uuid.UUID(content_hash[:32]))


class QdrantStore:
    """Vector store backed by Qdrant."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        dimension: int = 384,
        collection_suffix: str | None = None,
    ):
        self.client = QdrantClient(host=host, port=port)
        base = os.getenv("FUSESEARCH_COLLECTION", "fusesearch")
        if collection_suffix and collection_suffix != "local":
            self.collection_name = f"{base}-{collection_suffix}"
        else:
            self.collection_name = base
        self.dimension = dimension
        self._ensure_collection()

    def _ensure_collection(self):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
        self._ensure_text_index()

    def _ensure_text_index(self):
        """Create a full-text index on the content field for keyword search."""
        collection_info = self.client.get_collection(self.collection_name)
        if "content" not in (collection_info.payload_schema or {}):
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="content",
                field_schema=PayloadSchemaType.TEXT,
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]):
        """Insert or update chunks with their embeddings."""
        points = [
            PointStruct(
                id=hash_to_uuid(chunk.content_hash),
                vector=embedding,
                payload={
                    "content": chunk.content,
                    "title": chunk.title,
                    "source_type": chunk.source_type,
                    "document_source_id": chunk.document_source_id,
                    "heading_path": chunk.heading_path,
                    "chunk_index": chunk.chunk_index,
                    "url": chunk.url,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """Search for similar chunks by vector similarity."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        return [
            {
                "_id": point.id,
                "score": point.score,
                "content": point.payload["content"],
                "title": point.payload["title"],
                "source_type": point.payload["source_type"],
                "heading_path": point.payload["heading_path"],
                "metadata": point.payload["metadata"],
            }
            for point in results.points
        ]

    def keyword_search(self, query: str, limit: int = 20) -> list[dict]:
        """BM25 keyword search using Qdrant's full-text index."""
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="content", match=MatchText(text=query))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "_id": point.id,
                "content": point.payload["content"],
                "title": point.payload["title"],
                "source_type": point.payload["source_type"],
                "heading_path": point.payload["heading_path"],
                "metadata": point.payload["metadata"],
            }
            for point in results
        ]

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        limit: int = 5,
        vector_weight: float = 0.7,
    ) -> list[dict]:
        """Run vector + keyword search and fuse results with RRF."""
        vector_results = self.search(query_vector, limit=limit * 2)
        keyword_results = self.keyword_search(query_text, limit=limit * 2)
        fused = self._rrf_fuse(vector_results, keyword_results, vector_weight)
        return fused[:limit]

    @staticmethod
    def _rrf_fuse(
        vector_results: list[dict],
        keyword_results: list[dict],
        vector_weight: float = 0.7,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of two result lists."""
        keyword_weight = 1.0 - vector_weight
        scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, result in enumerate(vector_results):
            rid = str(result["_id"])
            scores[rid] = scores.get(rid, 0) + vector_weight / (k + rank + 1)
            result_map[rid] = result

        for rank, result in enumerate(keyword_results):
            rid = str(result["_id"])
            scores[rid] = scores.get(rid, 0) + keyword_weight / (k + rank + 1)
            if rid not in result_map:
                result_map[rid] = result

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{**result_map[rid], "score": score} for rid, score in ranked]

    def get_existing_hashes(self) -> set[str]:
        """Get all content hashes currently in the store."""
        hashes = set()
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            points, offset = result
            for point in points:
                hashes.add(point.id)
            if offset is None:
                break
        return hashes

    def delete_by_hashes(self, hashes: set[str]):
        """Delete chunks by their content hashes."""
        if not hashes:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=list(hashes),
        )

    def count(self) -> int:
        """Return number of indexed chunks."""
        return self.client.count(collection_name=self.collection_name).count
