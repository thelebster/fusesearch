import argparse
import os


def _make_embedder(provider: str | None = None):
    from fusesearch.core.embedder import create_embedder

    return create_embedder(provider)


def _make_store(embedder):
    from fusesearch.store.qdrant import QdrantStore

    return QdrantStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        dimension=embedder.dimension,
        collection_suffix=embedder.name,
    )


def cmd_serve(args):
    import uvicorn

    host = args.host or os.getenv("FUSESEARCH_HOST", "0.0.0.0")
    port = int(args.port or os.getenv("FUSESEARCH_PORT", "8000"))
    uvicorn.run("fusesearch.api.server:app", host=host, port=port)


def cmd_index(args):
    from fusesearch.indexer import Indexer
    from fusesearch.sources.local_files import LocalFilesAdapter

    embedder = _make_embedder(args.embedder)
    store = _make_store(embedder)

    adapter = LocalFilesAdapter(directories=args.paths)
    documents = list(adapter.fetch())
    print(f"Found {len(documents)} documents")

    indexer = Indexer(store=store, embedder=embedder)
    stats = indexer.index_documents(documents)
    print(
        f"Indexed: {stats['new']} new, {stats['skipped']} skipped, {stats['deleted']} deleted"
    )
    print(f"Total chunks in store: {store.count()}")


RERANK_OVERFETCH = 3


def cmd_search(args):
    embedder = _make_embedder(args.embedder)
    store = _make_store(embedder)
    hybrid = not args.no_hybrid

    fetch_limit = args.limit * RERANK_OVERFETCH if args.rerank else args.limit
    query_vector = embedder.embed_one(args.query)
    if hybrid:
        results = store.hybrid_search(query_vector, args.query, limit=fetch_limit)
    else:
        results = store.search(query_vector, limit=fetch_limit)

    if args.rerank:
        from fusesearch.core.reranker import create_reranker

        reranker = create_reranker()
        results = reranker.rerank(args.query, results, limit=args.limit)

    mode = "hybrid" if hybrid else "vector-only"
    rerank_label = " + rerank" if args.rerank else ""
    print(f"Search mode: {mode}{rerank_label}")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']:.4f}) ---")
        print(f"Title: {result['title']}")
        if result["heading_path"]:
            print(f"Section: {' > '.join(result['heading_path'])}")
        print(f"Content: {result['content'][:300]}...")


def main():
    parser = argparse.ArgumentParser(
        prog="fusesearch", description="FuseSearch - multi-source search"
    )
    parser.add_argument(
        "--embedder",
        choices=["local", "openai"],
        default=None,
        help="Embedding provider (default: FUSESEARCH_EMBEDDER env var or 'local')",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Serve command (default)
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default=None, help="Host to bind to")
    serve_parser.add_argument("--port", default=None, help="Port to bind to")

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index documents from local files"
    )
    index_parser.add_argument("paths", nargs="+", help="Directories to index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    hybrid_default = os.getenv("FUSESEARCH_HYBRID", "true").lower() == "true"
    search_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        default=not hybrid_default,
        help="Disable hybrid search (default: FUSESEARCH_HYBRID env var)",
    )
    rerank_default = os.getenv("FUSESEARCH_RERANK", "false").lower() == "true"
    search_parser.add_argument(
        "--rerank",
        action="store_true",
        default=rerank_default,
        help="Rerank results with cross-encoder (default: FUSESEARCH_RERANK env var)",
    )

    # MCP server command
    mcp_parser = subparsers.add_parser("mcp", help="Start the MCP server")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type",
    )

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "mcp":
        from fusesearch.mcp_server import main as mcp_main

        mcp_main(transport=args.transport)
    else:
        cmd_serve(argparse.Namespace(host=None, port=None))


if __name__ == "__main__":
    main()
