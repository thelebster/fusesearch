"""Configuration file support for FuseSearch.

Loads settings from fusesearch.yaml with env var interpolation.
Resolution order (highest wins): CLI args > env vars > config file > built-in defaults.
"""

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# Per-provider env vars for the embedding model (names predate the config file).
_EMBED_MODEL_ENV = {"local": "FUSESEARCH_EMBED_MODEL", "ollama": "OLLAMA_EMBED_MODEL"}


def _interpolate_env(value: str) -> str:
    """Replace ${VAR_NAME} placeholders with environment variable values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return _ENV_VAR_RE.sub(_replace, value)


def _interpolate_recursive(obj):
    """Walk nested dicts/lists, interpolating all string values."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _interpolate_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_recursive(item) for item in obj]
    return obj


# --- Pydantic settings models ---


def _none_to_empty(value):
    """An empty YAML section (``llm:`` with only comments below) loads as None;
    treat it like an empty mapping so the section's defaults apply."""
    return {} if value is None else value


class QdrantSettings(BaseModel):
    host: str = "localhost"
    port: int = 6333
    collection: str = "fusesearch"


class EmbedderSettings(BaseModel):
    provider: str = "local"
    # None = the provider's default model. Only applies while `provider` is in effect.
    model: str | None = None


class SearchSettings(BaseModel):
    hybrid: bool = True
    rerank: bool = False


class LLMSettings(BaseModel):
    provider: str | None = None


class DebugCacheSettings(BaseModel):
    enabled: bool = False
    directory: str = "./data/cache"


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class MCPSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001
    stateless: bool = True


class Settings(BaseModel):
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    debug_cache: DebugCacheSettings = Field(default_factory=DebugCacheSettings)
    api: APISettings = Field(default_factory=APISettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    @field_validator("*", mode="before")
    @classmethod
    def _empty_sections(cls, value):
        return _none_to_empty(value)


class SourceConfig(BaseModel):
    """A single source definition. Flat model — type determines which fields are used."""

    type: str
    # local_files
    paths: list[str] = Field(default_factory=list)
    # confluence
    url: str | None = None
    username: str | None = None
    api_token: str | None = None
    password: str | None = None
    spaces: list[str] = Field(default_factory=list)
    cloud: bool | None = None


class FuseSearchConfig(BaseModel):
    """Top-level config: settings + named sources."""

    settings: Settings = Field(default_factory=Settings)
    sources: dict[str, SourceConfig] = Field(default_factory=dict)

    @field_validator("*", mode="before")
    @classmethod
    def _empty_sections(cls, value):
        return _none_to_empty(value)


# --- ResolvedSettings: merges config file + env vars ---


class ResolvedSettings:
    """Resolved configuration that merges config file values with env var overrides.

    Resolution order (highest wins): env vars > config file > built-in defaults.
    CLI args are applied on top by the caller.
    """

    def __init__(self, config: FuseSearchConfig | None = None):
        self._config = config or FuseSearchConfig()

    @property
    def sources(self) -> dict[str, SourceConfig]:
        return self._config.sources

    # --- Qdrant ---

    @property
    def qdrant_host(self) -> str:
        return os.getenv("QDRANT_HOST") or self._config.settings.qdrant.host

    @property
    def qdrant_port(self) -> int:
        env = os.getenv("QDRANT_PORT")
        return int(env) if env else self._config.settings.qdrant.port

    @property
    def qdrant_collection(self) -> str:
        """Base collection name; the store appends the embedder name (except local)."""
        return (
            os.getenv("FUSESEARCH_COLLECTION")
            or self._config.settings.qdrant.collection
        )

    # --- Embedder ---

    @property
    def embedder_provider(self) -> str:
        return (
            os.getenv("FUSESEARCH_EMBEDDER") or self._config.settings.embedder.provider
        )

    @property
    def embedder_model(self) -> str | None:
        return self.embedder_model_for(self.embedder_provider)

    def embedder_model_for(self, provider: str) -> str | None:
        """Model for ``provider``: its env var, else the config model if the config
        was written for this provider, else None (provider default)."""
        env_name = _EMBED_MODEL_ENV.get(provider)
        if env_name and os.getenv(env_name):
            return os.getenv(env_name)
        embedder = self._config.settings.embedder
        return embedder.model if provider == embedder.provider else None

    # --- Search ---

    @property
    def hybrid(self) -> bool:
        env = os.getenv("FUSESEARCH_HYBRID")
        if env is not None:
            return env.lower() == "true"
        return self._config.settings.search.hybrid

    @property
    def rerank(self) -> bool:
        env = os.getenv("FUSESEARCH_RERANK")
        if env is not None:
            return env.lower() == "true"
        return self._config.settings.search.rerank

    # --- LLM ---

    @property
    def llm_provider(self) -> str | None:
        return os.getenv("FUSESEARCH_LLM") or self._config.settings.llm.provider

    # --- Debug cache ---

    @property
    def debug_cache_enabled(self) -> bool:
        env = os.getenv("FUSESEARCH_DEBUG_CACHE")
        if env is not None:
            return env.lower() == "true"
        return self._config.settings.debug_cache.enabled

    @property
    def debug_cache_dir(self) -> str:
        return (
            os.getenv("FUSESEARCH_CACHE_DIR")
            or self._config.settings.debug_cache.directory
        )

    # --- API ---

    @property
    def api_host(self) -> str:
        return os.getenv("FUSESEARCH_HOST") or self._config.settings.api.host

    @property
    def api_port(self) -> int:
        env = os.getenv("FUSESEARCH_PORT")
        return int(env) if env else self._config.settings.api.port

    # --- MCP ---

    @property
    def mcp_host(self) -> str:
        return os.getenv("MCP_HOST") or self._config.settings.mcp.host

    @property
    def mcp_port(self) -> int:
        env = os.getenv("MCP_PORT")
        return int(env) if env else self._config.settings.mcp.port

    @property
    def mcp_stateless(self) -> bool:
        env = os.getenv("MCP_STATELESS")
        if env is not None:
            return env.lower() == "true"
        return self._config.settings.mcp.stateless


# --- Config file discovery and loading ---


def find_config_file(explicit_path: str | None = None) -> Path | None:
    """Find config file: explicit path > FUSESEARCH_CONFIG env > auto-discover in CWD."""
    if explicit_path:
        path = Path(explicit_path)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Config file not found: {explicit_path}")

    env_path = os.getenv("FUSESEARCH_CONFIG")
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path
        raise FileNotFoundError(
            f"Config file from FUSESEARCH_CONFIG not found: {env_path}"
        )

    # Auto-discover in current directory
    for name in ("fusesearch.yaml", "fusesearch.yml"):
        path = Path(name)
        if path.is_file():
            return path

    return None


def load_config(config_path: str | None = None) -> FuseSearchConfig:
    """Load and validate config from YAML file.

    Returns default config if no config file exists.
    """
    path = find_config_file(config_path)
    if path is None:
        return FuseSearchConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not raw:
        return FuseSearchConfig()

    interpolated = _interpolate_recursive(raw)
    return FuseSearchConfig.model_validate(interpolated)


def load_settings(config_path: str | None = None) -> ResolvedSettings:
    """Load config and return resolved settings (config + env var overrides)."""
    config = load_config(config_path)
    return ResolvedSettings(config)
