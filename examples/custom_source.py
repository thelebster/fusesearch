"""Custom source adapter example: index documents from any data source.

This example shows how to implement a custom SourceAdapter to feed
documents into FuseSearch from sources beyond local files — APIs,
databases, CSV files, etc.

Prerequisites:
    - Qdrant running (docker compose up qdrant -d)
    - pip install fusesearch[local]

Usage:
    python examples/custom_source.py

Environment variables (optional):
    QDRANT_HOST  - Qdrant hostname (default: localhost)
    QDRANT_PORT  - Qdrant port (default: 6333)
"""

import os
from collections.abc import Iterator

from fusesearch import (
    Document,
    Indexer,
    QdrantStore,
    SourceAdapter,
    create_embedder,
)


class InMemorySource(SourceAdapter):
    """Example adapter that serves documents from an in-memory list.

    Replace this with your own data source: an API client, a database
    query, a CSV reader, etc. The only requirement is yielding Document
    objects from fetch().
    """

    def __init__(self, items: list[dict]):
        self.items = items

    @property
    def source_type(self) -> str:
        return "in_memory"

    def fetch(self) -> Iterator[Document]:
        for item in self.items:
            yield Document(
                source_type=self.source_type,
                source_id=item["id"],
                title=item["title"],
                content=item["content"],
                url=item.get("url"),
                metadata=item.get("metadata", {}),
            )

    def fetch_updated(self, since: str | None = None) -> Iterator[Document]:
        # For a real adapter, filter by modification date here.
        yield from self.fetch()


# --- Usage ---

# Your data, from anywhere
items = [
    {
        "id": "faq-1",
        "title": "Getting Started FAQ",
        "content": (
            "# Getting Started\n\n"
            "## How do I install FuseSearch?\n\n"
            "Install from PyPI: `pip install fusesearch[local]`.\n\n"
            "## What do I need to run it?\n\n"
            "You need Qdrant running. The easiest way is Docker:\n"
            "`docker run -p 6333:6333 qdrant/qdrant`"
        ),
    },
    {
        "id": "faq-2",
        "title": "Search Tips",
        "content": (
            "# Search Tips\n\n"
            "## How do I get better results?\n\n"
            "Use hybrid search (the default) for best recall. "
            "Enable reranking with `--rerank` for better precision on the top results.\n\n"
            "## What query length works best?\n\n"
            "Natural language questions work well. Short keywords also work "
            "thanks to the keyword search component of hybrid mode."
        ),
        "metadata": {"category": "tips"},
    },
]

# 1. Set up
embedder = create_embedder("local")
store = QdrantStore(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    dimension=embedder.dimension,
    collection_suffix=embedder.name,
)

# 2. Fetch from custom source
adapter = InMemorySource(items)
documents = list(adapter.fetch())
print(f"Documents from custom source: {len(documents)}")

# 3. Index
indexer = Indexer(store=store, embedder=embedder)
stats = indexer.index_documents(documents)
print(f"Indexed: {stats['new']} new, {stats['skipped']} unchanged")

# 4. Search
query = "How do I install FuseSearch?"
print(f"\nQuery: {query}\n")

query_vector = embedder.embed_one(query)
results = store.hybrid_search(query_vector, query, limit=3)

for i, result in enumerate(results, 1):
    print(f"--- Result {i} (score: {result['score']:.4f}) ---")
    print(f"Title: {result['title']}")
    print(f"Source: {result['source_type']}")
    print(f"{result['content'][:200]}...")
    print()
