"""Settings reach the CLI, store and embedder with CLI > env > config > default."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fusesearch.__main__ import _build_parser, _make_embedder, _pick
from fusesearch.config import load_config, load_settings
from fusesearch.store.qdrant import collection_name, create_store

ENV_VARS = [
    "FUSESEARCH_CONFIG",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "FUSESEARCH_COLLECTION",
    "QDRANT_COLLECTION",
    "FUSESEARCH_EMBEDDER",
    "FUSESEARCH_EMBED_MODEL",
    "OLLAMA_EMBED_MODEL",
    "FUSESEARCH_HYBRID",
    "FUSESEARCH_RERANK",
    "FUSESEARCH_LLM",
    "FUSESEARCH_DEBUG_CACHE",
    "FUSESEARCH_CACHE_DIR",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No env leaking in from .env / compose, no fusesearch.yaml auto-discovered."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _settings(tmp_path, yaml_text):
    path = tmp_path / "fusesearch.yaml"
    path.write_text(yaml_text)
    return load_settings(str(path))


# --- Embedder model ---

LOCAL_WITH_MODEL = """
settings:
  embedder:
    provider: local
    model: my-local-model
"""


def test_config_model_applies_to_its_provider(tmp_path):
    s = _settings(tmp_path, LOCAL_WITH_MODEL)
    assert s.embedder_model == "my-local-model"
    assert s.embedder_model_for("local") == "my-local-model"


def test_config_model_not_used_for_other_provider(tmp_path):
    s = _settings(tmp_path, LOCAL_WITH_MODEL)
    assert s.embedder_model_for("openai") is None
    assert s.embedder_model_for("ollama") is None


def test_env_provider_switch_drops_config_model(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSESEARCH_EMBEDDER", "openai")
    s = _settings(tmp_path, LOCAL_WITH_MODEL)
    assert s.embedder_provider == "openai"
    assert s.embedder_model is None


def test_model_env_vars_beat_config(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSESEARCH_EMBED_MODEL", "env-local-model")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "env-ollama-model")
    s = _settings(tmp_path, LOCAL_WITH_MODEL)
    assert s.embedder_model_for("local") == "env-local-model"
    assert s.embedder_model_for("ollama") == "env-ollama-model"


def test_make_embedder_passes_model_for_effective_provider(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "fusesearch.core.embedder.create_embedder",
        lambda provider, model=None: calls.append((provider, model)),
    )
    s = _settings(tmp_path, LOCAL_WITH_MODEL)

    _make_embedder(s)  # configured provider
    _make_embedder(s, "openai")  # --embedder openai

    assert calls == [("local", "my-local-model"), ("openai", None)]


# --- Collection ---


def test_collection_name_suffix_per_embedder():
    assert collection_name("fs", "local") == "fs"
    assert collection_name("fs", None) == "fs"
    assert collection_name("fs", "openai") == "fs-openai"


def test_collection_from_config_and_env(tmp_path, monkeypatch):
    yaml_text = "settings:\n  qdrant:\n    collection: from-config\n"
    assert _settings(tmp_path, yaml_text).qdrant_collection == "from-config"

    monkeypatch.setenv("FUSESEARCH_COLLECTION", "from-env")
    assert _settings(tmp_path, yaml_text).qdrant_collection == "from-env"


def test_create_store_uses_configured_collection(tmp_path, monkeypatch):
    monkeypatch.setattr("fusesearch.store.qdrant.QdrantClient", MagicMock())
    s = _settings(tmp_path, "settings:\n  qdrant:\n    collection: kb\n")
    embedder = MagicMock(dimension=1536)
    embedder.name = "openai"

    store = create_store(s, embedder)

    assert store.collection_name == "kb-openai"
    assert store.dimension == 1536


# --- CLI flags ---

NO_HYBRID_RERANK = """
settings:
  search:
    hybrid: false
    rerank: true
"""


def _search_opts(argv, settings):
    args = _build_parser().parse_args(argv)
    return _pick(args.hybrid, settings.hybrid), _pick(args.rerank, settings.rerank)


def test_cli_flags_unset_use_config(tmp_path):
    s = _settings(tmp_path, NO_HYBRID_RERANK)
    assert _search_opts(["search", "q"], s) == (False, True)
    assert _search_opts(["ask", "q"], s) == (False, True)


def test_cli_flags_override_config(tmp_path):
    s = _settings(tmp_path, NO_HYBRID_RERANK)
    assert _search_opts(["search", "q", "--hybrid", "--no-rerank"], s) == (True, False)


def test_legacy_no_hybrid_flag_still_works(tmp_path):
    s = _settings(tmp_path, "")
    assert _search_opts(["search", "q", "--no-hybrid"], s) == (False, False)


def test_env_beats_config_when_no_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSESEARCH_HYBRID", "true")
    monkeypatch.setenv("FUSESEARCH_RERANK", "false")
    s = _settings(tmp_path, NO_HYBRID_RERANK)
    assert _search_opts(["search", "q"], s) == (True, False)


def test_debug_cache_flags_resolve_against_config(tmp_path):
    s = _settings(
        tmp_path,
        "settings:\n  debug_cache:\n    enabled: true\n    directory: /tmp/dc\n",
    )
    args = _build_parser().parse_args(["index", "--all"])
    assert _pick(args.debug_cache, s.debug_cache_enabled) is True
    assert _pick(args.cache_dir, s.debug_cache_dir) == "/tmp/dc"

    args = _build_parser().parse_args(["index", "--all", "--no-debug-cache"])
    assert _pick(args.debug_cache, s.debug_cache_enabled) is False


def test_llm_provider_env_beats_config(tmp_path, monkeypatch):
    yaml_text = "settings:\n  llm:\n    provider: ollama\n"
    assert _settings(tmp_path, yaml_text).llm_provider == "ollama"
    monkeypatch.setenv("FUSESEARCH_LLM", "openai")
    assert _settings(tmp_path, yaml_text).llm_provider == "openai"


# --- Empty YAML sections ---


def test_empty_sections_use_defaults(tmp_path):
    s = _settings(tmp_path, "settings:\n  llm:\n  search:\nsources:\n")
    assert s.llm_provider is None
    assert s.hybrid is True
    assert s.sources == {}


def test_example_config_loads():
    example = Path(__file__).resolve().parents[1] / "fusesearch.example.yaml"
    if not example.is_file():
        pytest.skip("fusesearch.example.yaml not available (not mounted in Docker)")
    config = load_config(str(example))
    assert config.settings.llm.provider is None
    assert config.settings.embedder.model is None
    assert "blog-posts" in config.sources
