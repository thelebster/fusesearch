"""Basic search example: index local files and run hybrid search.

Prerequisites:
    - Qdrant running (docker compose up qdrant -d)
    - pip install fusesearch[local]

Usage:
    python examples/basic_search.py

Environment variables (optional):
    QDRANT_HOST  - Qdrant hostname (default: localhost)
    QDRANT_PORT  - Qdrant port (default: 6333)
"""

import os
from pathlib import Path

from fusesearch import Indexer, LocalFilesAdapter, QdrantStore, create_embedder

# 1. Create an embedder (local sentence-transformers by default)
embedder = create_embedder("local")

# 2. Connect to Qdrant and create/open a collection
store = QdrantStore(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    dimension=embedder.dimension,
    collection_suffix=embedder.name,
)

# 3. Fetch documents from local markdown files
docs_dir = Path(__file__).parent / "sample_docs"
adapter = LocalFilesAdapter(directories=[docs_dir])
documents = list(adapter.fetch())
print(f"Found {len(documents)} documents")

# 4. Index: chunk, embed, and store
indexer = Indexer(store=store, embedder=embedder)
stats = indexer.index_documents(documents)
print(f"Indexed: {stats['new']} new, {stats['skipped']} unchanged, {stats['deleted']} deleted")
print(f"Total chunks in store: {store.count()}")

# 5. Search
query = "How does hybrid search combine results?"
print(f"\nQuery: {query}\n")

query_vector = embedder.embed_one(query)
results = store.hybrid_search(query_vector, query, limit=3)

for i, result in enumerate(results, 1):
    print(f"--- Result {i} (score: {result['score']:.4f}) ---")
    print(f"Title: {result['title']}")
    if result["heading_path"]:
        print(f"Section: {' > '.join(result['heading_path'])}")
    print(f"{result['content'][:200]}...")
    print()
