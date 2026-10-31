import pytest

from fusesearch.config import (
    ConfluenceSource,
    FuseSearchConfig,
    LocalFilesSource,
    ResolvedSettings,
    _interpolate_env,
    _interpolate_recursive,
    find_config_file,
    load_config,
    load_settings,
)


class TestEnvInterpolation:
    def test_replaces_env_var(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "s3cret")
        assert _interpolate_env("token=${MY_SECRET}") == "token=s3cret"

    def test_leaves_unset_vars_unchanged(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        assert _interpolate_env("${NONEXISTENT_VAR}") == "${NONEXISTENT_VAR}"

    def test_multiple_vars_in_one_string(self, monkeypatch):
        monkeypatch.setenv("HOST", "localhost")
        monkeypatch.setenv("PORT", "6333")
        result = _interpolate_env("${HOST}:${PORT}")
        assert result == "localhost:6333"

    def test_no_vars_returns_unchanged(self):
        assert _interpolate_env("plain string") == "plain string"


class TestInterpolateRecursive:
    def test_walks_nested_dicts(self, monkeypatch):
        monkeypatch.setenv("DB_HOST", "myhost")
        data = {"qdrant": {"host": "${DB_HOST}"}}
        result = _interpolate_recursive(data)
        assert result == {"qdrant": {"host": "myhost"}}

    def test_walks_lists(self, monkeypatch):
        monkeypatch.setenv("PATH1", "/data/a")
        data = ["${PATH1}", "/data/b"]
        result = _interpolate_recursive(data)
        assert result == ["/data/a", "/data/b"]

    def test_preserves_non_strings(self):
        data = {"port": 6333, "enabled": True, "items": [1, 2]}
        result = _interpolate_recursive(data)
        assert result == data


class TestDefaultConfig:
    def test_empty_config_returns_defaults(self):
        config = FuseSearchConfig()
        assert config.settings.qdrant.host == "localhost"
        assert config.settings.qdrant.port == 6333
        assert config.settings.embedder.provider == "local"
        assert config.settings.search.hybrid is True
        assert config.settings.search.rerank is False
        assert config.sources == {}

    def test_default_resolved_settings(self, monkeypatch):
        # Clear env vars that could interfere
        for var in ("QDRANT_HOST", "QDRANT_PORT", "FUSESEARCH_EMBEDDER"):
            monkeypatch.delenv(var, raising=False)
        settings = ResolvedSettings()
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.embedder_provider == "local"
        assert settings.hybrid is True
        assert settings.rerank is False


class TestEnvVarOverrides:
    def test_env_overrides_config_value(self, monkeypatch):
        config = FuseSearchConfig()
        config.settings.qdrant.host = "from-config"
        monkeypatch.setenv("QDRANT_HOST", "from-env")
        settings = ResolvedSettings(config)
        assert settings.qdrant_host == "from-env"

    def test_env_overrides_port(self, monkeypatch):
        monkeypatch.setenv("QDRANT_PORT", "9999")
        settings = ResolvedSettings()
        assert settings.qdrant_port == 9999

    def test_env_overrides_hybrid(self, monkeypatch):
        config = FuseSearchConfig()
        config.settings.search.hybrid = True
        monkeypatch.setenv("FUSESEARCH_HYBRID", "false")
        settings = ResolvedSettings(config)
        assert settings.hybrid is False

    def test_env_overrides_rerank(self, monkeypatch):
        monkeypatch.setenv("FUSESEARCH_RERANK", "true")
        settings = ResolvedSettings()
        assert settings.rerank is True

    def test_config_value_used_when_no_env(self, monkeypatch):
        monkeypatch.delenv("QDRANT_HOST", raising=False)
        config = FuseSearchConfig()
        config.settings.qdrant.host = "config-host"
        settings = ResolvedSettings(config)
        assert settings.qdrant_host == "config-host"


class TestSourceConfig:
    def test_local_files_source(self):
        source = LocalFilesSource(
            type="local_files", paths=["/data/docs", "/data/blog"]
        )
        assert source.type == "local_files"
        assert source.paths == ["/data/docs", "/data/blog"]

    def test_confluence_source(self):
        source = ConfluenceSource(
            type="confluence",
            url="https://myorg.atlassian.net/wiki",
            username="user@example.com",
            api_token="secret",
            spaces=["DEV", "ENG"],
            cloud=True,
        )
        assert source.type == "confluence"
        assert source.url == "https://myorg.atlassian.net/wiki"
        assert source.spaces == ["DEV", "ENG"]

    def test_multiple_sources_in_config(self):
        config = FuseSearchConfig(
            sources={
                "blog": LocalFilesSource(type="local_files", paths=["/data/blog"]),
                "wiki": ConfluenceSource(
                    type="confluence",
                    url="https://myorg.atlassian.net/wiki",
                    spaces=["DEV"],
                ),
            }
        )
        assert len(config.sources) == 2
        assert "blog" in config.sources
        assert "wiki" in config.sources
        assert config.sources["blog"].type == "local_files"
        assert config.sources["wiki"].type == "confluence"


class TestConfigFileDiscovery:
    def test_explicit_path(self, tmp_path):
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("settings: {}")
        result = find_config_file(str(config_file))
        assert result == config_file

    def test_explicit_path_not_found(self):
        with pytest.raises(FileNotFoundError):
            find_config_file("/nonexistent/config.yaml")

    def test_env_var_path(self, tmp_path, monkeypatch):
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text("settings: {}")
        monkeypatch.setenv("FUSESEARCH_CONFIG", str(config_file))
        result = find_config_file()
        assert result == config_file

    def test_env_var_path_not_found(self, monkeypatch):
        monkeypatch.setenv("FUSESEARCH_CONFIG", "/nonexistent/config.yaml")
        with pytest.raises(FileNotFoundError):
            find_config_file()

    def test_auto_discover_yaml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text("settings: {}")
        result = find_config_file()
        assert result is not None
        assert result.resolve() == config_file.resolve()

    def test_auto_discover_yml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "fusesearch.yml"
        config_file.write_text("settings: {}")
        result = find_config_file()
        assert result is not None
        assert result.resolve() == config_file.resolve()

    def test_auto_discover_in_data_dir(self, tmp_path, monkeypatch):
        # Docker: ./data is mounted at /app/data and the working dir is /app.
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        config_file = tmp_path / "data" / "fusesearch.yaml"
        config_file.write_text("settings: {}")
        result = find_config_file()
        assert result is not None
        assert result.resolve() == config_file.resolve()

    def test_cwd_config_wins_over_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "fusesearch.yaml").write_text("settings: {}")
        cwd_file = tmp_path / "fusesearch.yaml"
        cwd_file.write_text("settings: {}")
        assert find_config_file().resolve() == cwd_file.resolve()

    def test_no_config_file_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        result = find_config_file()
        assert result is None


class TestLoadConfig:
    def test_load_from_yaml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text(
            """
settings:
  qdrant:
    host: custom-host
    port: 7777
sources:
  blog:
    type: local_files
    paths:
      - /data/blog
"""
        )
        config = load_config(str(config_file))
        assert config.settings.qdrant.host == "custom-host"
        assert config.settings.qdrant.port == 7777
        assert "blog" in config.sources
        assert config.sources["blog"].paths == ["/data/blog"]

    def test_load_with_env_interpolation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "interpolated-token")
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text(
            """
sources:
  wiki:
    type: confluence
    url: https://myorg.atlassian.net/wiki
    api_token: ${MY_TOKEN}
    spaces: [DEV]
"""
        )
        config = load_config(str(config_file))
        assert config.sources["wiki"].api_token == "interpolated-token"

    def test_load_empty_yaml(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text("")
        config = load_config(str(config_file))
        assert config.settings.qdrant.host == "localhost"
        assert config.sources == {}

    def test_load_no_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.settings.qdrant.host == "localhost"

    def test_partial_settings_filled_with_defaults(self, tmp_path):
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text(
            """
settings:
  qdrant:
    host: custom-host
"""
        )
        config = load_config(str(config_file))
        assert config.settings.qdrant.host == "custom-host"
        assert config.settings.qdrant.port == 6333  # default preserved
        assert config.settings.embedder.provider == "local"  # default preserved


class TestLoadSettings:
    def test_load_settings_returns_resolved(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QDRANT_HOST", raising=False)
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text(
            """
settings:
  qdrant:
    host: yaml-host
"""
        )
        settings = load_settings(str(config_file))
        assert isinstance(settings, ResolvedSettings)
        assert settings.qdrant_host == "yaml-host"

    def test_sources_accessible(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FUSESEARCH_CONFIG", raising=False)
        config_file = tmp_path / "fusesearch.yaml"
        config_file.write_text(
            """
sources:
  docs:
    type: local_files
    paths: [/data/docs]
"""
        )
        settings = load_settings(str(config_file))
        assert "docs" in settings.sources
        assert settings.sources["docs"].type == "local_files"
