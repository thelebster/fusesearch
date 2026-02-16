from fusesearch.core.chunker import chunk_document
from fusesearch.core.embedder import Embedder
from fusesearch.models import Chunk, Document
from fusesearch.store.qdrant import QdrantStore, hash_to_uuid

EMBED_BATCH_SIZE = 64


class Indexer:
    """Indexes documents into the vector store."""

    def __init__(self, store: QdrantStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def index_documents(self, documents: list[Document]) -> dict:
        """Chunk, diff, embed, and store documents.

        Returns stats: total_chunks, new, skipped, deleted.
        """
        # Chunk all documents
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(chunk_document(doc))

        # Diff against existing index
        new_ids = {hash_to_uuid(chunk.content_hash) for chunk in all_chunks}
        existing_ids = self.store.get_existing_hashes()

        to_add = [
            c for c in all_chunks if hash_to_uuid(c.content_hash) not in existing_ids
        ]
        to_delete = existing_ids - new_ids

        # Delete removed chunks
        self.store.delete_by_hashes(to_delete)

        # Embed and store new chunks in batches
        for i in range(0, len(to_add), EMBED_BATCH_SIZE):
            batch = to_add[i : i + EMBED_BATCH_SIZE]
            texts = [chunk.content for chunk in batch]
            embeddings = self.embedder.embed(texts)
            self.store.upsert(batch, embeddings)

        return {
            "total_chunks": len(all_chunks),
            "new": len(to_add),
            "skipped": len(all_chunks) - len(to_add),
            "deleted": len(to_delete),
        }
