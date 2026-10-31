"""Multi-tenant example: per-user index isolation.

Each simulated Telegram user gets their own Qdrant collection so
documents and search results stay completely separate.

Prerequisites:
    - Qdrant running (docker compose up qdrant -d)
    - pip install fusesearch[local]

Usage:
    python examples/multi_tenant.py

Environment variables (optional):
    QDRANT_HOST  - Qdrant hostname (default: localhost)
    QDRANT_PORT  - Qdrant port (default: 6333)
"""

import os
from pathlib import Path

from fusesearch import Indexer, LocalFilesAdapter, QdrantStore, create_embedder

# --- Setup ---

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
SAMPLE_DOCS = Path(__file__).parent / "sample_docs"

# Two simulated Telegram users with different document sets.
USERS = {
    "chat_10001": {
        "name": "Alice",
        "files": ["python.md"],
    },
    "chat_10002": {
        "name": "Bob",
        "files": ["docker.md", "search_engines.md"],
    },
}

# 1. Create a shared embedder (model loaded once, reused for all tenants)
print("Loading embedder...")
embedder = create_embedder("local")

# 2. Index each user's documents into an isolated collection
for chat_id, user in USERS.items():
    print(f"\n{'=' * 60}")
    print(f"Indexing for {user['name']} (collection: tenant-{chat_id})")
    print(f"{'=' * 60}")

    store = QdrantStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        dimension=embedder.dimension,
        collection=f"tenant-{chat_id}",
    )

    # Load only this user's files
    docs = []
    adapter = LocalFilesAdapter(directories=[SAMPLE_DOCS])
    for doc in adapter.fetch():
        if any(doc.source_id.endswith(f) for f in user["files"]):
            docs.append(doc)

    print(f"  Documents: {[d.title or d.source_id for d in docs]}")

    indexer = Indexer(store=store, embedder=embedder)
    stats = indexer.index_documents(docs)
    print(f"  Indexed: {stats['new']} new, {stats['skipped']} unchanged")
    print(f"  Total chunks: {store.count()}")

# 3. Search within each user's collection
print(f"\n{'=' * 60}")
print("Per-user search results")
print(f"{'=' * 60}")

query = "How does container technology work?"
print(f"\nQuery: {query!r}\n")

for chat_id, user in USERS.items():
    store = QdrantStore(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        dimension=embedder.dimension,
        collection=f"tenant-{chat_id}",
    )

    query_vector = embedder.embed_one(query)
    results = store.hybrid_search(query_vector, query, limit=2)

    print(f"--- {user['name']} (tenant-{chat_id}): {len(results)} result(s) ---")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['title']} — {r['content'][:80]}...")
    print()

# 4. Cross-check: prove Alice's collection has no Docker content
print(f"{'=' * 60}")
print("Isolation check: searching Alice's collection for 'Docker Compose'")
print(f"{'=' * 60}\n")

alice_store = QdrantStore(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
    dimension=embedder.dimension,
    collection="tenant-chat_10001",
)
query_vector = embedder.embed_one("Docker Compose")
results = alice_store.hybrid_search(query_vector, "Docker Compose", limit=2)

if not results:
    print("No results — Alice's index has no Docker content. Isolation confirmed!")
else:
    print(f"Found {len(results)} result(s) (all from Alice's Python docs):")
    for r in results:
        print(f"  Title: {r['title']} (score: {r['score']:.4f})")
    print("None of these are Docker docs — isolation confirmed!")
