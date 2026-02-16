# FuseSearch

Multi-source search aggregation tool that unifies retrieval across diverse data sources — Confluence, MCP servers, local files, and more — using AI-powered search and response synthesis through a single query interface.

## Quick Start

```bash
make build
make start
make index    # index docs from data/docs
make search "your query"
```

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
