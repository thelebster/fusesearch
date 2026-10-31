"""REST API with a config file: named sources, errors as 400, defaults from settings.

The server builds its embedder and store at import time, so the factories are
replaced with fakes before (re)importing it.
"""

import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeEmbedder, FakeStore

CONFIG = """
settings:
  search:
    rerank: true
sources:
  docs:
    type: local_files
    paths: [{docs}]
  wiki:
    type: confluence
    url: https://example.atlassian.net/wiki
    api_token: ${{WIKI_TOKEN}}
"""


@pytest.fixture
def api(monkeypatch, tmp_path):
    for name in ("FUSESEARCH_CONFIG", "FUSESEARCH_RERANK", "WIKI_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "rivers.md").write_text("# Rivers\n\nRivers flow into lakes.\n")
    (tmp_path / "fusesearch.yaml").write_text(CONFIG.format(docs=docs))

    store = FakeStore()
    monkeypatch.setattr(
        "fusesearch.core.embedder.create_embedder", lambda *a, **k: FakeEmbedder()
    )
    monkeypatch.setattr("fusesearch.store.qdrant.create_store", lambda s, e: store)
    sys.modules.pop("fusesearch.api.server", None)
    server = importlib.import_module("fusesearch.api.server")
    yield TestClient(server.app), store, server
    sys.modules.pop("fusesearch.api.server", None)


def test_index_named_source(api):
    client, store, _ = api
    resp = client.post("/index", json={"source": "docs"})

    assert resp.status_code == 200
    body = resp.json()
    assert (body["source"], body["documents_found"], body["new"]) == ("docs", 1, 1)
    assert store.sources() == ["docs"]


def test_index_source_with_unset_env_is_400(api):
    client, store, _ = api
    resp = client.post("/index", json={"source": "wiki"})

    assert resp.status_code == 400
    assert "WIKI_TOKEN" in resp.json()["detail"]
    assert store.points == {}


def test_index_unknown_adapter_type_is_400(api):
    client, _, _ = api
    resp = client.post("/index", json={"source": "dropbox"})

    assert resp.status_code == 400
    assert "dropbox" in resp.json()["detail"]


def test_search_defaults_come_from_config(api):
    client, _, server = api
    assert server.SearchRequest(query="x").rerank is True  # search.rerank: true
    assert server.SearchRequest(query="x").hybrid is True  # built-in default

    client.post("/index", json={"source": "docs"})
    resp = client.post("/search", json={"query": "rivers", "rerank": False})

    assert resp.status_code == 200
    assert resp.json()["results"][0]["source"] == "docs"


def test_health_counts_chunks(api):
    client, _, _ = api
    client.post("/index", json={"source": "docs"})
    assert client.get("/health").json() == {"status": "ok", "chunks_indexed": 1}
