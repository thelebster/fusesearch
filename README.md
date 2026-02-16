# FuseSearch

[![PyPI](https://img.shields.io/pypi/v/fusesearch)](https://pypi.org/project/fusesearch/)

Multi-source search aggregation tool that unifies retrieval across diverse data sources — Confluence, MCP servers, local files, and more — using AI-powered search and response synthesis through a single query interface.

## Installation

```bash
pip install fusesearch
```

With all optional dependencies (MCP server, local embeddings, OpenAI):

```bash
pip install fusesearch[all]
```

## Quick Start

```bash
make build
make start
make index    # index docs from data/docs
make search "your query"
```

## How It Works

```
Query → Embed → Vector Search  ─┐
                                ├→ RRF Fusion → Rerank (optional) → Results
                Keyword Search ─┘
```

## Embedding Providers

FuseSearch supports two embedding providers. Each uses a separate Qdrant collection due to different vector dimensions.

| | Local (default) | OpenAI |
|---|---|---|
| **Model** | all-MiniLM-L6-v2 (384 dims) | text-embedding-3-small (1536 dims) |
| **Quality** | Good for general English | Better for nuanced/complex queries |
| **Cost** | Free | ~$0.02 per 1M tokens |
| **Privacy** | Data stays local | Data sent to OpenAI |
| **Offline** | Yes | No |

### Local (default)

Uses [sentence-transformers](https://www.sbert.net/). Runs entirely on your machine, no API key needed.

### OpenAI

Uses [OpenAI](https://platform.openai.com/)'s API. Higher quality embeddings but requires an API key.

To use OpenAI embeddings, add to your `.env`:

```env
FUSESEARCH_EMBEDDER=openai
OPENAI_API_KEY=sk-...
```

Or pass via CLI:

```bash
fusesearch --embedder openai index data/docs
fusesearch --embedder openai search "your query"
```

**Rate limits:** OpenAI Tier 1 accounts have a 40k tokens-per-minute limit on embeddings. FuseSearch retries automatically on rate limit errors, but initial indexing of large document sets will be slow. Higher tiers (auto-upgrade as you spend) increase this significantly. See [OpenAI rate limits](https://platform.openai.com/docs/guides/rate-limits).

## Reranking

Reranking uses a [cross-encoder](https://www.sbert.net/docs/cross_encoder/pretrained_models.html) model to rescore search results after retrieval. The cross-encoder evaluates each (query, document) pair directly, producing more accurate relevance scores than initial retrieval alone.

When enabled, FuseSearch overfetches 3x candidates from hybrid search, then reranks down to the requested limit.

### Usage

Per-request via CLI flag or API parameter:

```bash
fusesearch search "your query" --rerank
```

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "your query", "rerank": true}'
```

Or enable globally via environment variable:

```env
FUSESEARCH_RERANK=true
```

### Configuration

| Variable | Default | Description |
|---|---|---|
| `FUSESEARCH_RERANK` | `false` | Enable reranking globally |
| `FUSESEARCH_RERANKER` | `local` | Reranker provider |
| `FUSESEARCH_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |

The reranker is independent of the embedding provider — it works on raw text, not vectors. You can use `FUSESEARCH_EMBEDDER=openai` with `FUSESEARCH_RERANK=true`. The local reranker requires the `[local]` extra (`sentence-transformers`).

## Ask (LLM Synthesis)

The `ask` command searches your indexed documents and uses an LLM to synthesize an answer with citations. This is optional — search works without any LLM provider installed.

### Usage

```bash
fusesearch ask "What is Drupal?"
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Drupal?"}'
```

### LLM Providers

| Provider | Model | Extra | Cost |
|---|---|---|---|
| Anthropic | claude-sonnet-4-20250514 | `[anthropic]` | Pay-as-you-go API |
| OpenAI | gpt-4o-mini | `[openai]` | Pay-as-you-go API |
| Ollama | llama3.2 | `[ollama]` | Free (runs locally) |

### Anthropic

Requires a separate API key (a Claude Pro/Team subscription does **not** include API access).

1. Create an account at [console.anthropic.com](https://console.anthropic.com/)
2. Add billing under **Settings > Billing** (pay-as-you-go)
3. Create a key under **Settings > API Keys**

```env
FUSESEARCH_LLM=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI

If you already have an API key for OpenAI embeddings, the same key works here.

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new secret key

```env
FUSESEARCH_LLM=openai
OPENAI_API_KEY=sk-...
```

### Ollama

No API key needed. Runs entirely on your machine.

1. Install from [ollama.com](https://ollama.com/)
2. Pull a model: `ollama pull llama3.2`

```env
FUSESEARCH_LLM=ollama
```

If `FUSESEARCH_LLM` is not set, FuseSearch auto-detects the first installed provider. If none are installed, `ask` returns a clear error — search continues to work normally.

## MCP Server

The `fusesearch-mcp` Docker service exposes a streamable HTTP endpoint on port 8001. Tools: `search` (hybrid search), `count` (indexed chunks).

### Claude Code

```bash
claude mcp add fusesearch http://localhost:8001/mcp --transport http
```

### Claude Desktop

**Option 1: Connectors UI (recommended)**

In Claude Desktop, go to **Settings > Connectors > Add custom connector** and enter `https://localhost:8001/mcp`.

**Option 2: Config file with `mcp-remote` bridge (local dev)**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fusesearch": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:8001/mcp", "--allow-http"]
    }
  }
}
```

Requires Node.js >= 18. `--allow-http` is required for plain HTTP (not needed for HTTPS).
