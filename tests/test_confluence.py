from unittest.mock import MagicMock, patch

import pytest

from fusesearch.sources.confluence import ConfluenceAdapter, _storage_to_markdown


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
        assert "/wiki/spaces/DEV/pages/12345" in doc.url

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


class TestFetch:
    def _make_adapter_with_mock_client(self, pages_by_space=None):
        adapter = ConfluenceAdapter(
            url="https://myorg.atlassian.net/wiki",
            spaces=["DEV"],
        )
        mock_client = MagicMock()

        if pages_by_space is None:
            pages_by_space = {
                "DEV": [
                    {
                        "id": "1",
                        "title": "Page 1",
                        "body": {"storage": {"value": "<p>Content 1.</p>"}},
                        "version": {"number": 1, "when": "2024-01-01"},
                        "ancestors": [],
                    },
                    {
                        "id": "2",
                        "title": "Page 2",
                        "body": {"storage": {"value": "<p>Content 2.</p>"}},
                        "version": {"number": 1, "when": "2024-01-02"},
                        "ancestors": [],
                    },
                ],
            }

        def get_all_pages(space_key, start=0, limit=50, expand=""):
            pages = pages_by_space.get(space_key, [])
            return pages[start : start + limit]

        mock_client.get_all_pages_from_space.side_effect = get_all_pages
        adapter._client = mock_client
        return adapter, mock_client

    def test_fetches_all_pages(self):
        adapter, _ = self._make_adapter_with_mock_client()
        docs = list(adapter.fetch())
        assert len(docs) == 2
        assert docs[0].title == "Page 1"
        assert docs[1].title == "Page 2"

    def test_iterates_configured_spaces(self):
        adapter = ConfluenceAdapter(
            url="https://myorg.atlassian.net/wiki",
            spaces=["DEV", "OPS"],
        )
        mock_client = MagicMock()
        mock_client.get_all_pages_from_space.return_value = []
        adapter._client = mock_client

        list(adapter.fetch())

        called_spaces = [
            call.kwargs.get("space_key", call.args[0] if call.args else None)
            for call in mock_client.get_all_pages_from_space.call_args_list
        ]
        assert "DEV" in called_spaces
        assert "OPS" in called_spaces

    def test_pagination(self):
        # Create 60 pages to trigger pagination (limit=50)
        pages = [
            {
                "id": str(i),
                "title": f"Page {i}",
                "body": {"storage": {"value": f"<p>Content {i}.</p>"}},
                "version": {"number": 1, "when": "2024-01-01"},
                "ancestors": [],
            }
            for i in range(60)
        ]
        adapter, mock_client = self._make_adapter_with_mock_client(
            pages_by_space={"DEV": pages}
        )

        docs = list(adapter.fetch())
        assert len(docs) == 60
        # Should have made 2 calls: 50 + 10
        assert mock_client.get_all_pages_from_space.call_count == 2

    def test_discovers_spaces_when_not_configured(self):
        adapter = ConfluenceAdapter(
            url="https://myorg.atlassian.net/wiki",
            spaces=[],
        )
        mock_client = MagicMock()
        mock_client.get_all_spaces.return_value = {
            "results": [{"key": "AUTO1"}, {"key": "AUTO2"}],
        }
        mock_client.get_all_pages_from_space.return_value = []
        adapter._client = mock_client

        list(adapter.fetch())

        mock_client.get_all_spaces.assert_called_once()
        called_spaces = [
            call.args[0] for call in mock_client.get_all_pages_from_space.call_args_list
        ]
        assert called_spaces == ["AUTO1", "AUTO2"]


class TestFetchUpdated:
    def test_falls_back_to_full_fetch_when_since_is_none(self):
        adapter = ConfluenceAdapter(
            url="https://myorg.atlassian.net/wiki",
            spaces=["DEV"],
        )
        mock_client = MagicMock()
        mock_client.get_all_pages_from_space.return_value = []
        adapter._client = mock_client

        list(adapter.fetch_updated(since=None))

        mock_client.get_all_pages_from_space.assert_called()
        mock_client.cql.assert_not_called()

    def test_uses_cql_with_since(self):
        adapter = ConfluenceAdapter(
            url="https://myorg.atlassian.net/wiki",
            spaces=["DEV"],
        )
        mock_client = MagicMock()
        mock_client.cql.return_value = {"results": []}
        adapter._client = mock_client

        list(adapter.fetch_updated(since="2024-06-15T10:30:00"))

        cql_call = mock_client.cql.call_args
        cql_query = cql_call.args[0] if cql_call.args else cql_call.kwargs.get("cql")
        assert 'space = "DEV"' in cql_query
        assert 'lastmodified >= "2024-06-15 10:30"' in cql_query

    def test_yields_documents_from_cql_results(self):
        adapter = ConfluenceAdapter(
            url="https://myorg.atlassian.net/wiki",
            spaces=["DEV"],
        )
        mock_client = MagicMock()
        mock_client.cql.return_value = {
            "results": [
                {
                    "content": {
                        "id": "99",
                        "title": "Updated Page",
                        "body": {"storage": {"value": "<p>New content.</p>"}},
                        "version": {"number": 5, "when": "2024-06-16"},
                        "ancestors": [],
                    },
                },
            ],
        }
        adapter._client = mock_client

        docs = list(adapter.fetch_updated(since="2024-06-15T00:00:00"))
        assert len(docs) == 1
        assert docs[0].title == "Updated Page"
        assert docs[0].source_id == "confluence:DEV:99"
