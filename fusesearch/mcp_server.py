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
    # Stateless mode: each request is independent, no persistent sessions.
    # Prevents 404 errors when clients reconnect after server restarts (e.g. Docker rebuild).
    # Disable if you need server-to-client notifications or server-initiated sampling.
    # https://gofastmcp.com/python-sdk/fastmcp-server-http
    stateless_http=os.getenv("MCP_STATELESS", "true").lower() == "true",
)

# Lazy-initialized globals
_embedder = None
_store = None
_reranker = None

RERANK_OVERFETCH = 3
_hybrid_default = os.getenv("FUSESEARCH_HYBRID", "true").lower() == "true"
_rerank_default = os.getenv("FUSESEARCH_RERANK", "false").lower() == "true"


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fusesearch.core.embedder import create_embedder

        _embedder = create_embedder()
    return _embedder


def _get_store():
    global _store
    if _store is None:
        from fusesearch.store.qdrant import QdrantStore

        embedder = _get_embedder()
        _store = QdrantStore(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
            dimension=embedder.dimension,
            collection_suffix=embedder.name,
        )
    return _store


def _get_reranker():
    global _reranker
    if _reranker is None:
        from fusesearch.core.reranker import create_reranker

        _reranker = create_reranker()
    return _reranker


@mcp.tool()
def search(
    query: str,
    limit: int = 5,
    hybrid: bool = _hybrid_default,
    rerank: bool = _rerank_default,
) -> str:
    """Search the FuseSearch knowledge base. Use this tool whenever the user asks a factual or knowledge question — about a topic, concept, person, event, or anything that indexed documents might answer. Returns relevant document chunks with source titles and scores. Always search BEFORE answering knowledge questions."""
    embedder = _get_embedder()
    store = _get_store()

    fetch_limit = limit * RERANK_OVERFETCH if rerank else limit
    query_vector = embedder.embed_one(query)
    if hybrid:
        results = store.hybrid_search(query_vector, query, limit=fetch_limit)
    else:
        results = store.search(query_vector, limit=fetch_limit)
    if rerank:
        reranker = _get_reranker()
        results = reranker.rerank(query, results, limit=limit)

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


@mcp.prompt()
def ask(question: str) -> str:
    """Ask a question about the indexed knowledge base. Use the search tool to find relevant documents, then synthesize an answer with citations."""
    return (
        f"Use the FuseSearch search tool to answer this question: {question}\n\n"
        "Instructions:\n"
        "1. Search for relevant documents using the search tool (try multiple queries if needed)\n"
        "2. Synthesize a clear, concise answer based on the search results\n"
        "3. Cite sources using [1], [2], etc. matching the result numbers\n"
        "4. If the results don't contain enough information, say so"
    )


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
