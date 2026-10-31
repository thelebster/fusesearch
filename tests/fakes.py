"""In-memory stand-ins for QdrantStore and Embedder (no Qdrant, no model download)."""

from fusesearch.store.qdrant import point_id


class FakeEmbedder:
    dimension = 3
    name = "fake"

    def embed(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_one(self, text):
        return [0.1, 0.2, 0.3]


class FakeStore:
    """Point id -> chunk, scoped by ``chunk.source`` like the real store."""

    collection_name = "fake"

    def __init__(self):
        self.points = {}

    # --- QdrantStore API used by Indexer, CLI and API ---

    def get_existing_hashes(self, source=None):
        return {
            pid
            for pid, chunk in self.points.items()
            if source is None or chunk.source == source
        }

    def delete_by_hashes(self, ids):
        for pid in ids:
            self.points.pop(pid, None)

    def upsert(self, chunks, embeddings):
        for chunk in chunks:
            self.points[point_id(chunk.source, chunk.content_hash)] = chunk

    def count(self, source=None):
        return len(self.get_existing_hashes(source))

    def count_unscoped(self):
        return 0

    def search(self, query_vector, limit=5):
        return self._results(limit)

    def hybrid_search(self, query_vector, query_text, limit=5, vector_weight=0.7):
        return self._results(limit)

    def _results(self, limit):
        return [
            {
                "_id": pid,
                "score": 1.0,
                "content": c.content,
                "title": c.title,
                "source": c.source,
                "source_type": c.source_type,
                "heading_path": c.heading_path,
                "metadata": c.metadata,
            }
            for pid, c in list(self.points.items())[:limit]
        ]

    # --- test helpers ---

    def sources(self):
        return sorted({c.source for c in self.points.values()})

    def contents(self, source):
        return sorted(c.content for c in self.points.values() if c.source == source)
