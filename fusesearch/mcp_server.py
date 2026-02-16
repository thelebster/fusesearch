import os
from typing import Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "FuseSearch",
    instructions=(
        "FuseSearch is a knowledge base with indexed documents, blog posts, and notes. "
        "Use this server when the user asks factual questions, wants to look up a topic, "
        "or needs information that might exist in indexed sources."
    ),
    host=os.getenv("MCP_HOST", "0.0.0.0"),
    port=int(os.getenv("MCP_PORT", "8001")),
)

# Lazy-initialized globals
_embedder = None
_store = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fusesearch.core.embedder import LocalEmbedder

        _embedder = LocalEmbedder()
    return _embedder


def _get_store():
    global _store
    if _store is None:
        from fusesearch.store.qdrant import QdrantStore

        _store = QdrantStore(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            dimension=_get_embedder().dimension,
        )
    return _store


@mcp.tool()
def search(query: str, limit: int = 5) -> str:
    """Search the FuseSearch knowledge base. Use this tool whenever the user asks a factual or knowledge question — about a topic, concept, person, event, or anything that indexed documents might answer. Returns relevant document chunks with source titles and scores. Always search BEFORE answering knowledge questions."""
    embedder = _get_embedder()
    store = _get_store()

    query_vector = embedder.embed_one(query)
    results = store.hybrid_search(query_vector, query, limit=limit)

    if not results:
        return "No results found."

    parts = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        heading = ""
        if result.get("heading_path"):
            heading = f" > {' > '.join(result['heading_path'])}"
        score = result.get("score", 0)
        content = result.get("content", "")
        parts.append(f"[{i}] {title}{heading} (score: {score:.4f})\n{content}")

    return "\n\n---\n\n".join(parts)


@mcp.tool()
def count() -> str:
    """Return the number of indexed chunks in the store."""
    store = _get_store()
    return f"{store.count()} chunks indexed"


Transport = Literal["stdio", "sse", "streamable-http"]


def main(transport: Transport = "streamable-http"):
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
