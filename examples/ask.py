"""Ask example: search + LLM synthesis with citations.

Prerequisites:
    - Qdrant running (docker compose up qdrant -d)
    - pip install fusesearch[local]
    - An LLM provider installed and configured, e.g.:
        pip install fusesearch[anthropic]  # and set ANTHROPIC_API_KEY
        pip install fusesearch[openai]     # and set OPENAI_API_KEY
        pip install fusesearch[ollama]     # and run: ollama pull llama3.2

Usage:
    python examples/ask.py

Environment variables (optional):
    QDRANT_HOST  - Qdrant hostname (default: localhost)
    QDRANT_PORT  - Qdrant port (default: 6333)
"""

import os
from pathlib import Path

from fusesearch import (
    Indexer,
    LocalFilesAdapter,
    QdrantStore,
    create_embedder,
    create_llm,
    synthesize,
)

# 1. Set up embedder and store
embedder = create_embedder("local")
store = QdrantStore(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
    dimension=embedder.dimension,
    collection_suffix=embedder.name,
)

# 2. Index sample docs (skips unchanged chunks on re-run)
docs_dir = Path(__file__).parent / "sample_docs"
adapter = LocalFilesAdapter(directories=[docs_dir])
documents = list(adapter.fetch())

indexer = Indexer(store=store, embedder=embedder)
stats = indexer.index_documents(documents)
print(f"Indexed: {stats['new']} new, {stats['skipped']} unchanged")

# 3. Search
query = "What is the difference between vector search and keyword search?"
print(f"\nQuestion: {query}\n")

query_vector = embedder.embed_one(query)
results = store.hybrid_search(query_vector, query, limit=5)

# 4. Create LLM (auto-detects first available provider)
llm = create_llm()

# 5. Synthesize answer with citations
response = synthesize(llm, query, results)

print(response["answer"])
print()
if response["sources"]:
    print("Sources:")
    for source in response["sources"]:
        heading = ""
        if source["heading_path"]:
            heading = f" > {' > '.join(source['heading_path'])}"
        print(f"  [{source['index']}] {source['title']}{heading}")
