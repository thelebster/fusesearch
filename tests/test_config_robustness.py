"""Config validation, unset ${VAR} handling, and `index` source selection."""

import sys

import pytest

from fusesearch import __main__ as cli
from fusesearch.config import ConfigError, load_config, load_settings
from tests.fakes import FakeEmbedder, FakeStore

ENV_VARS = [
    "FUSESEARCH_CONFIG",
    "FUSESEARCH_EMBEDDER",
    "FUSESEARCH_DEBUG_CACHE",
    "CONFLUENCE_URL",
    "CONFLUENCE_USERNAME",
    "CONFLUENCE_API_TOKEN",
    "CONFLUENCE_SPACES",
    "WIKI_TOKEN",
    "QDRANT_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _write(tmp_path, text):
    path = tmp_path / "fusesearch.yaml"
    path.write_text(text)
    return str(path)


# --- Validation ---


@pytest.mark.parametrize(
    ("yaml_text", "expected"),
    [
        ("settings:\n  serach:\n    hybrid: false\n", "serach"),
        (
            "sources:\n  wiki:\n    type: confluence\n    url: https://x\n    space: [DEV]\n",
            "space",
        ),
        ("sources:\n  docs:\n    type: local_files\n", "paths"),
        ("sources:\n  docs:\n    type: local_files\n    paths: []\n", "paths"),
        ("sources:\n  docs:\n    type: dropbox\n", "dropbox"),
        ("settings: [1, 2]\n", "settings"),
    ],
)
def test_invalid_config_raises_readable_error(tmp_path, yaml_text, expected):
    path = _write(tmp_path, yaml_text)
    with pytest.raises(ConfigError, match="is invalid") as exc:
        load_config(path)
    assert expected in str(exc.value)
    assert path in str(exc.value)


def test_invalid_yaml_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(_write(tmp_path, "settings: [unclosed\n"))


def test_top_level_must_be_mapping(tmp_path):
    with pytest.raises(ConfigError, match="mapping"):
        load_config(_write(tmp_path, "- a\n- b\n"))


def test_confluence_spaces_default_is_none(tmp_path):
    config = load_config(
        _write(
            tmp_path, "sources:\n  wiki:\n    type: confluence\n    url: https://x\n"
        )
    )
    # None (not []) so the adapter still falls back to CONFLUENCE_SPACES.
    assert config.sources["wiki"].spaces is None


# --- Unset ${VAR} ---

TWO_SOURCES = """
sources:
  docs:
    type: local_files
    paths: [{docs}]
  wiki:
    type: confluence
    url: https://example.atlassian.net/wiki
    username: me@example.com
    api_token: ${{WIKI_TOKEN}}
"""


def test_unset_var_in_settings_fails_at_load(tmp_path):
    path = _write(tmp_path, "settings:\n  qdrant:\n    host: ${QDRANT_KEY}\n")
    with pytest.raises(ConfigError, match="QDRANT_KEY"):
        load_config(path)


def test_unset_var_in_source_loads_and_fails_only_that_source(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    settings = load_settings(_write(tmp_path, TWO_SOURCES.format(docs=docs)))

    assert settings.sources["wiki"].missing_env == ["WIKI_TOKEN"]
    assert settings.sources["docs"].missing_env == []
    with pytest.raises(ConfigError, match="'wiki'.*WIKI_TOKEN"):
        settings.create_source_adapter("wiki")
    assert settings.create_source_adapter("docs").source_type == "local_files"


def test_set_var_in_source_is_interpolated(tmp_path, monkeypatch):
    monkeypatch.setenv("WIKI_TOKEN", "s3cret")
    settings = load_settings(_write(tmp_path, TWO_SOURCES.format(docs=tmp_path)))
    assert settings.sources["wiki"].api_token == "s3cret"
    assert settings.sources["wiki"].missing_env == []


# --- `fusesearch index` source selection ---


def _index_args(*argv):
    return cli._build_parser().parse_args(["index", *argv])


def _settings_with_sources(tmp_path):
    return load_settings(_write(tmp_path, TWO_SOURCES.format(docs=tmp_path)))


def test_index_without_args_requires_all(tmp_path):
    settings = _settings_with_sources(tmp_path)
    with pytest.raises(cli.UsageError, match="--all") as exc:
        cli._resolve_sources(_index_args(), settings)
    assert "docs, wiki" in str(exc.value)


def test_index_all_without_config_sources(tmp_path):
    with pytest.raises(cli.UsageError, match="No sources defined"):
        cli._resolve_sources(_index_args("--all"), load_settings())


def test_unknown_source_lists_choices(tmp_path):
    settings = _settings_with_sources(tmp_path)
    with pytest.raises(cli.UsageError, match="Unknown source 'nope'") as exc:
        cli._resolve_sources(_index_args("--source", "nope"), settings)
    assert "docs, wiki" in str(exc.value)
    assert "confluence" in str(exc.value)


def test_index_all_binds_each_source_name(tmp_path, monkeypatch):
    settings = _settings_with_sources(tmp_path)
    created = []
    monkeypatch.setattr(settings, "create_source_adapter", created.append)

    sources = cli._resolve_sources(_index_args("--all"), settings)
    for _, make_adapter in sources:
        make_adapter()

    assert [name for name, _ in sources] == ["docs", "wiki"]
    assert created == ["docs", "wiki"]  # not ["wiki", "wiki"]


def test_index_all_skips_broken_source_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# A\n\nSome text.\n")
    path = _write(tmp_path, TWO_SOURCES.format(docs=docs))
    store = FakeStore()
    monkeypatch.setattr(cli, "_make_embedder", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(cli, "_make_store", lambda *a, **k: store)

    args = cli._build_parser().parse_args(["--config", path, "index", "--all"])
    with pytest.raises(SystemExit) as exc:
        cli.cmd_index(args)

    assert exc.value.code == 1
    assert store.sources() == ["docs"]  # docs indexed despite wiki failing
    err = capsys.readouterr().err
    assert "Source 'wiki'" in err
    assert "(skipped)" in err
    assert "WIKI_TOKEN" in err


def test_main_prints_usage_error_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fusesearch", "index"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert capsys.readouterr().err.startswith("Error: Nothing to index")


def test_main_prints_config_error(tmp_path, monkeypatch, capsys):
    path = _write(tmp_path, "settings:\n  serach: {}\n")
    monkeypatch.setattr(sys, "argv", ["fusesearch", "--config", path, "index", "--all"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "serach" in capsys.readouterr().err
