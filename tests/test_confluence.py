from unittest.mock import MagicMock, patch

import pytest
import requests

from fusesearch.sources.confluence import (
    ConfluenceAdapter,
    ConfluenceError,
    _storage_to_markdown,
)


@pytest.fixture(autouse=True)
def _no_confluence_env(monkeypatch):
    """CONFLUENCE_* from a developer .env (passed into Docker) must not leak in."""
    for name in (
        "CONFLUENCE_URL",
        "CONFLUENCE_USERNAME",
        "CONFLUENCE_API_TOKEN",
        "CONFLUENCE_PASSWORD",
        "CONFLUENCE_SPACES",
        "CONFLUENCE_CLOUD",
    ):
        monkeypatch.delenv(name, raising=False)


class TestStorageToMarkdown:
    def test_basic_html(self):
        html = "<h1>Title</h1><p>Paragraph text.</p>"
        md = _storage_to_markdown(html)
        assert "# Title" in md
        assert "Paragraph text." in md

    def test_strips_ac_elements(self):
        html = (
            '<ac:structured-macro ac:name="toc">'
            "<ac:parameter>maxLevel=3</ac:parameter>"
            "</ac:structured-macro>"
            "<p>Content after macro.</p>"
        )
        md = _storage_to_markdown(html)
        assert "ac:" not in md
        assert "Content after macro." in md

    def test_strips_ri_elements(self):
        html = (
            '<ac:link><ri:page ri:content-title="Other Page"/></ac:link>'
            "<p>Some text.</p>"
        )
        md = _storage_to_markdown(html)
        assert "ri:" not in md
        assert "Some text." in md

    def test_converts_tables(self):
        html = "<table><tr><th>Header</th></tr><tr><td>Cell value</td></tr></table>"
        md = _storage_to_markdown(html)
        assert "Header" in md
        assert "Cell value" in md

    def test_converts_links(self):
        html = '<p>Visit <a href="https://example.com">Example</a>.</p>'
        md = _storage_to_markdown(html)
        assert "Example" in md
        assert "https://example.com" in md

    def test_empty_input(self):
        assert _storage_to_markdown("") == ""

    def test_nested_ac_elements(self):
        html = (
            '<ac:structured-macro ac:name="panel">'
            "<ac:rich-text-body>"
            "<p>Panel content.</p>"
            "</ac:rich-text-body>"
            "</ac:structured-macro>"
        )
        md = _storage_to_markdown(html)
        assert "ac:" not in md
        assert "Panel content." in md


class TestConfluenceAdapterConstruction:
    def test_requires_url(self):
        with pytest.raises(ValueError, match="Confluence URL is required"):
            ConfluenceAdapter()

    def test_source_type(self):
        adapter = ConfluenceAdapter(url="https://example.atlassian.net/wiki")
        assert adapter.source_type == "confluence"

    def test_auto_detects_cloud_from_url(self):
        adapter = ConfluenceAdapter(url="https://myorg.atlassian.net/wiki")
        assert adapter.cloud is True

    def test_auto_detects_server_from_url(self):
        adapter = ConfluenceAdapter(url="https://confluence.example.com")
        assert adapter.cloud is False

    def test_explicit_cloud_overrides_detection(self):
        adapter = ConfluenceAdapter(url="https://confluence.example.com", cloud=True)
        assert adapter.cloud is True

    def test_spaces_from_args(self):
        adapter = ConfluenceAdapter(
            url="https://example.atlassian.net/wiki",
            spaces=["DEV", "OPS"],
        )
        assert adapter.spaces == ["DEV", "OPS"]

    def test_spaces_from_env(self):
        with patch.dict("os.environ", {"CONFLUENCE_SPACES": "HR, FINANCE"}):
            adapter = ConfluenceAdapter(url="https://example.atlassian.net/wiki")
            assert adapter.spaces == ["HR", "FINANCE"]

    def test_url_trailing_slash_stripped(self):
        adapter = ConfluenceAdapter(url="https://example.atlassian.net/wiki/")
        assert adapter.url == "https://example.atlassian.net/wiki"

    def test_url_from_env(self):
        with patch.dict("os.environ", {"CONFLUENCE_URL": "https://env.example.com"}):
            adapter = ConfluenceAdapter()
            assert adapter.url == "https://env.example.com"


class TestPageToDocument:
    def _make_adapter(self):
        return ConfluenceAdapter(url="https://myorg.atlassian.net/wiki")

    def _make_page(self, **overrides):
        page = {
            "id": "12345",
            "title": "Test Page",
            "body": {
                "storage": {"value": "<p>Hello world.</p>"},
            },
            "version": {"number": 3, "when": "2024-06-15T10:30:00.000Z"},
            "ancestors": [
                {"id": "100", "title": "Parent Page"},
            ],
        }
        page.update(overrides)
        return page

    def test_converts_page_to_document(self):
        adapter = self._make_adapter()
        doc = adapter._page_to_document(self._make_page(), "DEV")
        assert doc is not None
        assert doc.source_id == "confluence:DEV:12345"
        assert doc.title == "Test Page"
        assert "Hello world." in doc.content
        assert doc.source_type == "confluence"
        assert doc.metadata["space_key"] == "DEV"
        assert doc.metadata["page_id"] == "12345"
        assert doc.metadata["version"] == 3
        assert doc.metadata["ancestors"] == [{"id": "100", "title": "Parent Page"}]

    def test_cloud_url_format(self):
        adapter = self._make_adapter()
        doc = adapter._page_to_document(self._make_page(), "DEV")
        assert doc.url == "https://myorg.atlassian.net/wiki/spaces/DEV/pages/12345"

    def test_server_url_format(self):
        adapter = ConfluenceAdapter(url="https://confluence.example.com", cloud=False)
        doc = adapter._page_to_document(self._make_page(), "DEV")
        assert "pageId=12345" in doc.url

    def test_skips_empty_body(self):
        adapter = self._make_adapter()
        page = self._make_page(body={"storage": {"value": ""}})
        doc = adapter._page_to_document(page, "DEV")
        assert doc is None

    def test_skips_missing_body(self):
        adapter = self._make_adapter()
        page = self._make_page(body={})
        doc = adapter._page_to_document(page, "DEV")
        assert doc is None


class FakeClient:
    """Answers client.get(url, params=..., absolute=True) from a URL -> responses map.

    Each value is a list of responses served in order (for paging).
    """

    def __init__(self, routes):
        self.routes = {url: list(responses) for url, responses in routes.items()}
        self.calls = []

    def get(self, url, params=None, absolute=False):
        assert absolute, "adapter must request absolute URLs"
        self.calls.append((url, params))
        path = url.split("?")[0]
        if path not in self.routes or not self.routes[path]:
            raise AssertionError(f"unexpected request: {url}")
        return self.routes[path].pop(0)


SITE = "https://myorg.atlassian.net"
V2 = f"{SITE}/wiki/api/v2"


def _v2_page(
    page_id, title, body="<p>Text.</p>", parent=None, created="2024-06-15T10:30:00.000Z"
):
    return {
        "id": page_id,
        "title": title,
        "parentId": parent,
        "version": {"number": 2, "createdAt": created},
        "body": {"storage": {"value": body, "representation": "storage"}},
        "_links": {"webui": f"/spaces/DEV/pages/{page_id}/{title.replace(' ', '+')}"},
    }


def _cloud_adapter(routes, **kwargs):
    adapter = ConfluenceAdapter(url=f"{SITE}/wiki", spaces=["DEV"], **kwargs)
    adapter._client = FakeClient(routes)
    return adapter


SPACES_DEV = {"results": [{"id": "77", "key": "DEV"}], "_links": {}}


class TestCloudV2Fetch:
    def test_fetches_pages_with_urls_and_ancestors(self):
        pages = {
            "results": [
                _v2_page("1", "Root"),
                _v2_page("2", "Child", parent="1"),
                _v2_page("3", "Grandchild", parent="2"),
            ],
            "_links": {"base": f"{SITE}/wiki"},
        }
        adapter = _cloud_adapter(
            {f"{V2}/spaces": [SPACES_DEV], f"{V2}/spaces/77/pages": [pages]}
        )

        docs = list(adapter.fetch())

        assert [d.source_id for d in docs] == [
            "confluence:DEV:1",
            "confluence:DEV:2",
            "confluence:DEV:3",
        ]
        # One /wiki, not /wiki/wiki
        assert docs[2].url == f"{SITE}/wiki/spaces/DEV/pages/3/Grandchild"
        assert docs[2].metadata["ancestors"] == [
            {"id": "1", "title": "Root"},
            {"id": "2", "title": "Child"},
        ]
        assert docs[2].metadata["modified_at"] == "2024-06-15T10:30:00.000Z"
        assert docs[0].metadata["ancestors"] == []

    def test_requests_storage_body_and_space_keys(self):
        adapter = _cloud_adapter(
            {
                f"{V2}/spaces": [SPACES_DEV],
                f"{V2}/spaces/77/pages": [{"results": [], "_links": {}}],
            }
        )
        list(adapter.fetch())

        (_, spaces_params), (_, pages_params) = adapter.client.calls
        assert spaces_params["keys"] == "DEV"
        assert pages_params["body-format"] == "storage"

    def test_follows_cursor_next_links_from_site_root(self):
        first = {
            "results": [_v2_page("1", "One")],
            "_links": {"next": "/wiki/api/v2/spaces/77/pages?cursor=abc"},
        }
        second = {"results": [_v2_page("2", "Two")], "_links": {}}
        adapter = _cloud_adapter(
            {f"{V2}/spaces": [SPACES_DEV], f"{V2}/spaces/77/pages": [first, second]}
        )

        docs = list(adapter.fetch())

        assert [d.title for d in docs] == ["One", "Two"]
        assert adapter.client.calls[-1] == (
            f"{V2}/spaces/77/pages?cursor=abc",
            None,
        )

    def test_unknown_space_key_is_an_error(self):
        adapter = _cloud_adapter({f"{V2}/spaces": [{"results": [], "_links": {}}]})
        with pytest.raises(ConfluenceError, match="not found or not visible: DEV"):
            list(adapter.fetch())

    def test_discovers_all_spaces_when_none_configured(self):
        adapter = ConfluenceAdapter(url=SITE)  # site URL without /wiki also works
        adapter.spaces = []
        adapter._client = FakeClient(
            {
                f"{V2}/spaces": [
                    {
                        "results": [
                            {"id": "77", "key": "DEV"},
                            {"id": "88", "key": "OPS"},
                        ]
                    }
                ],
                f"{V2}/spaces/77/pages": [{"results": [_v2_page("1", "A")]}],
                f"{V2}/spaces/88/pages": [{"results": [_v2_page("2", "B")]}],
            }
        )

        docs = list(adapter.fetch())

        assert [d.source_id for d in docs] == ["confluence:DEV:1", "confluence:OPS:2"]
        assert "keys" not in adapter.client.calls[0][1]

    def test_fetch_updated_filters_on_version_created_at(self):
        pages = {
            "results": [
                _v2_page("1", "Old", created="2024-01-01T00:00:00.000Z"),
                _v2_page("2", "New", created="2024-06-15T10:30:00.000Z"),
            ]
        }
        adapter = _cloud_adapter(
            {f"{V2}/spaces": [SPACES_DEV], f"{V2}/spaces/77/pages": [pages]}
        )

        docs = list(adapter.fetch_updated(since="2024-06-01T00:00:00"))

        assert [d.title for d in docs] == ["New"]

    def test_skips_pages_without_body(self):
        pages = {"results": [_v2_page("1", "Empty", body="")]}
        adapter = _cloud_adapter(
            {f"{V2}/spaces": [SPACES_DEV], f"{V2}/spaces/77/pages": [pages]}
        )
        assert list(adapter.fetch()) == []


class TestErrors:
    def _http_error(self, status):
        response = requests.Response()
        response.status_code = status
        return requests.HTTPError("rejected", response=response)

    def test_rejected_credentials_explain_tokens(self):
        adapter = ConfluenceAdapter(url=f"{SITE}/wiki", spaces=["DEV"])
        adapter._client = MagicMock()
        adapter._client.get.side_effect = self._http_error(403)

        with pytest.raises(ConfluenceError, match=r"\(403\).*API tokens expire"):
            list(adapter.fetch())

    def test_non_json_response_is_an_error(self):
        # e.g. a redirect to the login page answered with an HTML/empty body
        adapter = ConfluenceAdapter(url=f"{SITE}/wiki", spaces=["DEV"])
        adapter._client = MagicMock()
        adapter._client.get.return_value = None

        with pytest.raises(ConfluenceError, match="no JSON"):
            list(adapter.fetch())

    def test_api_gateway_url_counts_as_cloud(self):
        adapter = ConfluenceAdapter(
            url="https://api.atlassian.com/ex/confluence/abc-123"
        )
        assert adapter.cloud is True
        assert adapter._v2("spaces") == (
            "https://api.atlassian.com/ex/confluence/abc-123/wiki/api/v2/spaces"
        )


SERVER = "https://confluence.example.com"


def _server_adapter(routes):
    adapter = ConfluenceAdapter(url=SERVER, spaces=["DEV"], cloud=False)
    adapter._client = FakeClient(routes)
    return adapter


def _v1_page(page_id, title):
    return {
        "id": page_id,
        "title": title,
        "body": {"storage": {"value": "<p>Text.</p>"}},
        "version": {"number": 1, "when": "2024-06-15T10:30:00.000Z"},
        "ancestors": [{"id": "9", "title": "Parent"}],
    }


class TestServerV1Fetch:
    def test_fetches_pages_and_follows_next_relative_to_base(self):
        first = {
            "results": [_v1_page("1", "One")],
            "_links": {
                "base": SERVER,
                "next": "/rest/api/content?spaceKey=DEV&start=50",
            },
        }
        second = {"results": [_v1_page("2", "Two")], "_links": {"base": SERVER}}
        adapter = _server_adapter({f"{SERVER}/rest/api/content": [first, second]})

        docs = list(adapter.fetch())

        assert [d.title for d in docs] == ["One", "Two"]
        assert docs[0].url == f"{SERVER}/pages/viewpage.action?pageId=1"
        assert docs[0].metadata["ancestors"] == [{"id": "9", "title": "Parent"}]
        params = adapter.client.calls[0][1]
        assert params["spaceKey"] == "DEV"
        assert params["expand"] == "body.storage,version,ancestors"

    def test_fetch_updated_uses_cql(self):
        adapter = _server_adapter(
            {
                f"{SERVER}/rest/api/content/search": [
                    {"results": [_v1_page("1", "Changed")], "_links": {}}
                ]
            }
        )

        docs = list(adapter.fetch_updated(since="2024-06-15T10:30:00"))

        assert [d.title for d in docs] == ["Changed"]
        cql = adapter.client.calls[0][1]["cql"]
        assert 'space = "DEV"' in cql
        assert 'lastmodified >= "2024-06-15 10:30"' in cql

    def test_fetch_updated_none_falls_back_to_full_fetch(self):
        adapter = _server_adapter(
            {f"{SERVER}/rest/api/content": [{"results": [_v1_page("1", "A")]}]}
        )
        assert [d.title for d in adapter.fetch_updated(since=None)] == ["A"]
