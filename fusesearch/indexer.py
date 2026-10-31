from tqdm import tqdm

from fusesearch.core.chunker import chunk_document
from fusesearch.core.embedder import Embedder
from fusesearch.models import Chunk, Document
from fusesearch.store.qdrant import QdrantStore, point_id

EMBED_BATCH_SIZE = 64


class Indexer:
    """Indexes documents into the vector store."""

    def __init__(
        self, store: QdrantStore, embedder: Embedder, cache_dir: str | None = None
    ):
        self.store = store
        self.embedder = embedder
        self.cache_dir = cache_dir

    def index_documents(
        self,
        documents: list[Document],
        source: str | None = None,
        incremental: bool = False,
    ) -> dict:
        """Chunk, diff, embed, and store documents of one source.

        Args:
            documents: All documents of the source (full sync), or only the changed
                ones (``incremental=True``).
            source: Index namespace, e.g. the config source name. Diffing and
                deletion only look at chunks of this source. Defaults to the
                documents' ``source_type`` (all documents must share it).
            incremental: Only add new chunks, never delete. Use when ``documents``
                is a subset, e.g. fetched with ``fetch_updated(since=...)``.

        Returns stats: source, total_chunks, new, skipped, deleted.
        """
        source = _resolve_source(documents, source)

        # Debug cache: dump pre-chunking documents to disk
        if self.cache_dir:
            from fusesearch.core.debug_cache import dump_documents

            n = dump_documents(documents, self.cache_dir)
            print(f"Debug cache: wrote {n} files to {self.cache_dir}")

        # Chunk all documents
        all_chunks: list[Chunk] = []
        for doc in tqdm(documents, desc="Chunking", unit="doc"):
            all_chunks.extend(chunk_document(doc))
        for chunk in all_chunks:
            chunk.source = source

        # Diff against the existing chunks of this source only
        stats = {"source": source, "total_chunks": len(all_chunks)}
        if source is None:  # no documents and no source name: nothing to diff
            return {**stats, "new": 0, "skipped": 0, "deleted": 0}

        new_ids = {point_id(source, c.content_hash) for c in all_chunks}
        existing_ids = self.store.get_existing_hashes(source=source)

        to_add = [
            c
            for c in all_chunks
            if point_id(source, c.content_hash) not in existing_ids
        ]
        to_delete = set() if incremental else existing_ids - new_ids

        # Guard: a full sync that fetched nothing would wipe the source. That's far
        # more often an outage or bad credentials than a really emptied source.
        if not documents and to_delete:
            print(
                f"Source {source!r}: 0 documents fetched; keeping its "
                f"{len(to_delete)} existing chunks. Check the source if that's wrong."
            )
            to_delete = set()

        # Delete removed chunks
        if to_delete:
            self.store.delete_by_hashes(to_delete)

        # Embed and store new chunks in batches
        if to_add:
            pbar = tqdm(total=len(to_add), desc="Embedding", unit="chunk")
            for i in range(0, len(to_add), EMBED_BATCH_SIZE):
                batch = to_add[i : i + EMBED_BATCH_SIZE]
                texts = [chunk.content for chunk in batch]
                embeddings = self.embedder.embed(texts)
                self.store.upsert(batch, embeddings)
                pbar.update(len(batch))
            pbar.close()

        return {
            **stats,
            "new": len(to_add),
            "skipped": len(all_chunks) - len(to_add),
            "deleted": len(to_delete),
        }


def _resolve_source(documents: list[Document], source: str | None) -> str | None:
    if source is not None:
        return source
    types = {doc.source_type for doc in documents}
    if len(types) > 1:
        raise ValueError(
            f"Documents have several source types {sorted(types)}; pass source= "
            "so each source is indexed (and cleaned up) separately."
        )
    return types.pop() if types else None
