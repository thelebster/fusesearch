import argparse
import os
import sys

from fusesearch.config import ConfigError


def _load_settings(args):
    """Load config file + env vars into ResolvedSettings."""
    from fusesearch.config import load_settings

    config_path = getattr(args, "config", None)
    return load_settings(config_path)


def _pick(cli_value, setting):
    """CLI value if the flag was given, else the resolved setting (env > config)."""
    return setting if cli_value is None else cli_value


def _make_embedder(settings, provider: str | None = None):
    """Embedder for --embedder (if given) or the configured provider."""
    from fusesearch.core.embedder import create_embedder

    provider = provider or settings.embedder_provider
    return create_embedder(provider, model=settings.embedder_model_for(provider))


def _make_store(embedder, settings):
    from fusesearch.store.qdrant import create_store

    return create_store(settings, embedder)


ADAPTER_TYPES = ("local_files", "confluence")


class UsageError(Exception):
    """Bad command-line usage; printed without a traceback."""


def _resolve_sources(args, settings):
    """Which sources to index, from CLI args and config.

    Returns (name, make_adapter) pairs. Adapters are created lazily, inside the
    per-source loop, so one broken source doesn't stop the others with --all.
    """
    from fusesearch.sources import create_adapter

    source_arg = getattr(args, "source", None)
    paths = getattr(args, "paths", [])
    index_all = getattr(args, "all", False)

    # 1. Legacy: positional paths
    if paths:
        return [
            ("local_files", lambda: create_adapter("local_files", {"paths": paths}))
        ]

    # 2. --source: named source from config, else an adapter type (legacy)
    if source_arg:
        if source_arg in settings.sources:
            return [(source_arg, lambda: settings.create_source_adapter(source_arg))]
        if source_arg == "confluence":
            spaces = getattr(args, "spaces", None)
            return [
                ("confluence", lambda: create_adapter("confluence", {"spaces": spaces}))
            ]
        if source_arg == "local_files":
            raise UsageError(
                "--source local_files needs paths: fusesearch index PATH..."
            )
        known = ", ".join(settings.sources) or "none"
        raise UsageError(
            f"Unknown source {source_arg!r}. Config sources: {known}. "
            f"Adapter types: {', '.join(ADAPTER_TYPES)}."
        )

    # 3. --all: every source in the config file
    if index_all:
        if not settings.sources:
            raise UsageError(
                "No sources defined in the config file (see fusesearch.example.yaml)."
            )
        return [
            (name, lambda name=name: settings.create_source_adapter(name))
            for name in settings.sources
        ]

    # 4. Nothing given: don't guess (a full re-index can be slow and costly)
    hint = (
        f" Config sources: {', '.join(settings.sources)}." if settings.sources else ""
    )
    raise UsageError("Nothing to index. Give paths, --source NAME, or --all." + hint)


def cmd_serve(args):
    import uvicorn

    settings = _load_settings(args)
    host = args.host or settings.api_host
    port = int(args.port) if args.port else settings.api_port

    # Pass config path to API server via env so it picks up the same config
    config_path = getattr(args, "config", None)
    if config_path:
        os.environ["FUSESEARCH_CONFIG"] = config_path

    uvicorn.run("fusesearch.api.server:app", host=host, port=port)


def cmd_index(args):
    from fusesearch.indexer import Indexer

    settings = _load_settings(args)
    sources = _resolve_sources(args, settings)  # fail fast, before loading models
    embedder = _make_embedder(settings, args.embedder)
    store = _make_store(embedder, settings)

    since = getattr(args, "since", None)
    debug_cache = _pick(args.debug_cache, settings.debug_cache_enabled)
    cache_dir = _pick(args.cache_dir, settings.debug_cache_dir) if debug_cache else None
    indexer = Indexer(store=store, embedder=embedder, cache_dir=cache_dir)
    if since:
        print(f"Incremental run (--since {since}): adding/updating only, no deletions")

    total_docs = 0
    total_stats = {"new": 0, "skipped": 0, "deleted": 0}
    failed = []

    for name, make_adapter in sources:
        if len(sources) > 1:
            print(f"\n--- Source: {name} ---")

        try:
            adapter = make_adapter()
            if since:
                documents = list(adapter.fetch_updated(since=since))
            else:
                documents = list(adapter.fetch())
            print(f"Found {len(documents)} documents")
            stats = indexer.index_documents(
                documents, source=name, incremental=bool(since)
            )
        except Exception as e:  # noqa: BLE001 - one broken source must not stop the others
            sys.stdout.flush()  # keep stdout/stderr in order when piped
            if isinstance(e, ConfigError):
                message = f"{e} (skipped)"
            else:
                message = f"source {name!r} skipped: {type(e).__name__}: {e}"
            print(f"Error: {message}", file=sys.stderr)
            failed.append(name)
            continue

        print(
            f"Indexed: {stats['new']} new, {stats['skipped']} skipped, "
            f"{stats['deleted']} deleted"
        )
        total_docs += len(documents)
        for key in total_stats:
            total_stats[key] += stats[key]

    if len(sources) > 1:
        print("\n--- Total ---")
        print(
            f"Found {total_docs} documents across {len(sources) - len(failed)} sources"
        )
        print(
            f"Indexed: {total_stats['new']} new, {total_stats['skipped']} skipped, "
            f"{total_stats['deleted']} deleted"
        )

    print(f"Total chunks in store: {store.count()}")
    if failed:
        sys.stdout.flush()
        print(f"Failed sources: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


RERANK_OVERFETCH = 3


def cmd_search(args):
    settings = _load_settings(args)
    embedder = _make_embedder(settings, args.embedder)
    store = _make_store(embedder, settings)
    hybrid = _pick(args.hybrid, settings.hybrid)
    rerank = _pick(args.rerank, settings.rerank)

    results = _run_search(store, embedder, args.query, args.limit, hybrid, rerank)

    mode = "hybrid" if hybrid else "vector-only"
    rerank_label = " + rerank" if rerank else ""
    print(f"Search mode: {mode}{rerank_label}")
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {result['score']:.4f}) ---")
        print(f"Title: {result['title']}")
        if result["heading_path"]:
            print(f"Section: {' > '.join(result['heading_path'])}")
        print(f"Content: {result['content'][:300]}...")


def _run_search(store, embedder, query, limit, hybrid, rerank):
    """Shared search logic for cmd_search and cmd_ask."""
    fetch_limit = limit * RERANK_OVERFETCH if rerank else limit
    query_vector = embedder.embed_one(query)
    if hybrid:
        results = store.hybrid_search(query_vector, query, limit=fetch_limit)
    else:
        results = store.search(query_vector, limit=fetch_limit)
    if rerank:
        from fusesearch.core.reranker import create_reranker

        reranker = create_reranker()
        results = reranker.rerank(query, results, limit=limit)
    return results


def cmd_ask(args):
    from fusesearch.core.synthesizer import synthesize
    from fusesearch.llm import create_llm

    settings = _load_settings(args)
    embedder = _make_embedder(settings, args.embedder)
    store = _make_store(embedder, settings)
    hybrid = _pick(args.hybrid, settings.hybrid)
    rerank = _pick(args.rerank, settings.rerank)

    results = _run_search(store, embedder, args.query, args.limit, hybrid, rerank)

    try:
        llm = create_llm(args.llm or settings.llm_provider)
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    response = synthesize(llm, args.query, results)

    print(f"\n{response['answer']}\n")
    if response["sources"]:
        print("Sources:")
        for source in response["sources"]:
            heading = ""
            if source["heading_path"]:
                heading = f" > {' > '.join(source['heading_path'])}"
            print(f"  [{source['index']}] {source['title']}{heading}")


def _add_search_flags(parser):
    """--hybrid/--no-hybrid and --rerank/--no-rerank. Default None: resolved after
    loading settings, so CLI > env > config file > built-in default."""
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hybrid vector + keyword search "
        "(default: FUSESEARCH_HYBRID, config search.hybrid, or on)",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rerank results with cross-encoder "
        "(default: FUSESEARCH_RERANK, config search.rerank, or off)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fusesearch", description="FuseSearch - multi-source search"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to fusesearch.yaml config file",
    )
    parser.add_argument(
        "--embedder",
        choices=["local", "openai", "ollama"],
        default=None,
        help="Embedding provider "
        "(default: FUSESEARCH_EMBEDDER, config embedder.provider, or 'local')",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Serve command (default)
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default=None, help="Host to bind to")
    serve_parser.add_argument("--port", default=None, help="Port to bind to")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents from a source")
    index_parser.add_argument(
        "paths", nargs="*", default=[], help="Directories to index (for local_files)"
    )
    index_parser.add_argument(
        "--source",
        default=None,
        help="Source name from config or adapter type (local_files, confluence)",
    )
    index_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        default=False,
        help="Index all sources defined in config file",
    )
    index_parser.add_argument(
        "--spaces",
        nargs="*",
        default=None,
        help="Confluence spaces to index (default: all or CONFLUENCE_SPACES env var)",
    )
    index_parser.add_argument(
        "--since",
        default=None,
        help="Only index documents updated since this ISO datetime (incremental sync)",
    )
    index_parser.add_argument(
        "--debug-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Dump pre-chunking documents to disk "
        "(default: FUSESEARCH_DEBUG_CACHE, config debug_cache.enabled, or off)",
    )
    index_parser.add_argument(
        "--cache-dir",
        default=None,
        help="Debug cache directory "
        "(default: FUSESEARCH_CACHE_DIR, config debug_cache.directory, or ./data/cache)",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    _add_search_flags(search_parser)

    # Ask command
    ask_parser = subparsers.add_parser(
        "ask", help="Ask a question and get a synthesized answer"
    )
    ask_parser.add_argument("query", help="Question to ask")
    ask_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to use"
    )
    _add_search_flags(ask_parser)
    ask_parser.add_argument(
        "--llm",
        choices=["anthropic", "openai", "ollama"],
        default=None,
        help="LLM provider (default: FUSESEARCH_LLM, config llm.provider, "
        "or the first installed provider)",
    )

    # MCP server command
    mcp_parser = subparsers.add_parser("mcp", help="Start the MCP server")
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    try:
        _dispatch(args)
    except (ConfigError, UsageError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


def _dispatch(args):
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "ask":
        cmd_ask(args)
    elif args.command == "mcp":
        # Pass config path to MCP server via env
        config_path = getattr(args, "config", None)
        if config_path:
            os.environ["FUSESEARCH_CONFIG"] = config_path

        from fusesearch.mcp_server import main as mcp_main

        mcp_main(transport=args.transport)
    else:
        cmd_serve(argparse.Namespace(host=None, port=None, config=args.config))


if __name__ == "__main__":
    main()
