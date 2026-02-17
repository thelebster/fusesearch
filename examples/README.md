# FuseSearch Examples

Programmatic usage examples for the FuseSearch Python library.

## Quick Start (Docker)

Run any example with a single command — no local Python or Qdrant setup needed. Each example runs in an isolated environment with its own Qdrant instance.

```bash
make example-basic
make example-ask
make example-custom
```

Cleanup when done:

```bash
make example-clean
```

## Running Locally

If you prefer running outside Docker:

1. **Qdrant** must be running:

   ```bash
   docker run -d -p 6333:6333 qdrant/qdrant
   ```

2. **Install FuseSearch** with local embeddings:

   ```bash
   pip install fusesearch[local]
   ```

3. Run an example:

   ```bash
   python examples/basic_search.py
   ```

**Warning:** When running locally against a shared Qdrant instance, examples will modify the default `fusesearch` collection. Set `FUSESEARCH_COLLECTION=examples` to use a separate collection and protect your main index.

## Examples

### basic_search.py

Index local markdown files and run hybrid search. This is the minimal "hello world" for FuseSearch.

Shows: `create_embedder` → `QdrantStore` → `LocalFilesAdapter` → `Indexer` → `hybrid_search`

### ask.py

Search + LLM-powered answer synthesis with citations. Requires an LLM API key passed from the host:

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  make example-ask

# or
OPENAI_API_KEY=sk-... \
  FUSESEARCH_LLM=openai \
  make example-ask
```

Shows: everything in basic_search + `create_llm` → `synthesize`

### custom_source.py

Implement a custom `SourceAdapter` to index documents from any data source (API, database, CSV, etc.).

Shows: subclass `SourceAdapter` → implement `fetch()` → index and search custom documents
