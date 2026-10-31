import os
import uuid
from hashlib import sha256

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchText,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from fusesearch.models import Chunk


def hash_to_uuid(content_hash: str) -> str:
    """Convert a SHA-256 hex string to a UUID (uses first 32 hex chars)."""
    return str(uuid.UUID(content_hash[:32]))


def point_id(source: str, content_hash: str) -> str:
    """Point ID for a chunk: unique per (source, content), so the same text indexed
    from two sources is stored twice and deleting it from one leaves the other."""
    return hash_to_uuid(sha256(f"{source}\0{content_hash}".encode()).hexdigest())


def _source_filter(source: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


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
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self):
        """Full-text index on content (keyword search), keyword index on source
        (per-source diffing and deletion)."""
        schema = self.client.get_collection(self.collection_name).payload_schema or {}
        wanted = {
            "content": PayloadSchemaType.TEXT,
            "source": PayloadSchemaType.KEYWORD,
        }
        for field, field_schema in wanted.items():
            if field not in schema:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=field_schema,
                )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]):
        """Insert or update chunks with their embeddings."""
        points = [
            PointStruct(
                id=point_id(chunk.source, chunk.content_hash),
                vector=embedding,
                payload={
                    "source": chunk.source,
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
                "source": point.payload.get("source"),
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
                "source": point.payload.get("source"),
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

    def get_existing_hashes(self, source: str | None = None) -> set[str]:
        """Get the point IDs currently in the store.

        With ``source``, only points of that source; without, the whole collection.
        """
        hashes = set()
        offset = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=_source_filter(source) if source is not None else None,
                limit=1000,
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

    def count(self, source: str | None = None) -> int:
        """Return number of indexed chunks, optionally for one source."""
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=_source_filter(source) if source is not None else None,
        ).count

    def count_unscoped(self) -> int:
        """Chunks without a ``source`` field, written by fusesearch <= 0.2.x.

        They are never diffed or deleted by source-scoped indexing, so they linger as
        duplicates; drop the collection and re-index to get rid of them.
        """
        return self.client.count(
            collection_name=self.collection_name,
            count_filter=Filter(
                must=[IsEmptyCondition(is_empty=PayloadField(key="source"))]
            ),
        ).count
