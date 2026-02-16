import argparse
import os


def _make_embedder():
    from fusesearch.core.embedder import LocalEmbedder

    return LocalEmbedder()


def _make_store(embedder):
    from fusesearch.store.qdrant import QdrantStore

    return QdrantStore(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        dimension=embedder.dimension,
    )


def cmd_serve(args):
    import uvicorn

    host = args.host or os.getenv("FUSESEARCH_HOST", "0.0.0.0")
    port = int(args.port or os.getenv("FUSESEARCH_PORT", "8000"))
    uvicorn.run("fusesearch.api.server:app", host=host, port=port)


def cmd_index(args):
    from fusesearch.indexer import Indexer
    from fusesearch.sources.local_files import LocalFilesAdapter

    embedder = _make_embedder()
    store = _make_store(embedder)

    adapter = LocalFilesAdapter(directories=args.paths)
    documents = list(adapter.fetch())
    print(f"Found {len(documents)} documents")

    indexer = Indexer(store=store, embedder=embedder)
    stats = indexer.index_documents(documents)
    print(f"Indexed: {stats['new']} new, {stats['skipped']} skipped, {stats['deleted']} deleted")
    print(f"Total chunks in store: {store.count()}")


def cmd_search(args):
    embedder = _make_embedder()
    store = _make_store(embedder)
    hybrid = not args.no_hybrid

    query_vector = embedder.embed_one(args.query)
    if hybrid:
        results = store.hybrid_search(query_vector, args.query, limit=args.limit)
    else:
        results = store.search(query_vector, limit=args.limit)

    mode = "hybrid" if hybrid else "vector-only"
    print(f"Search mode: {mode}")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']:.4f}) ---")
        print(f"Title: {result['title']}")
        if result["heading_path"]:
            print(f"Section: {' > '.join(result['heading_path'])}")
        print(f"Content: {result['content'][:300]}...")


def main():
    parser = argparse.ArgumentParser(prog="fusesearch", description="FuseSearch - multi-source search")
    subparsers = parser.add_subparsers(dest="command")

    # Serve command (default)
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default=None, help="Host to bind to")
    serve_parser.add_argument("--port", default=None, help="Port to bind to")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents from local files")
    index_parser.add_argument("paths", nargs="+", help="Directories to index")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    search_parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search (vector-only)")

    # MCP server command
    mcp_parser = subparsers.add_parser("mcp", help="Start the MCP server")
    mcp_parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default="stdio", help="Transport type")

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
