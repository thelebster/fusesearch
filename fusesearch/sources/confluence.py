import logging
import os
import re
from collections.abc import Iterator

from fusesearch.models import Document
from fusesearch.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

# Regex to strip Confluence-specific XML namespace elements (ac:*, ri:*)
_AC_TAG_RE = re.compile(r"</?ac:[^>]*>", re.DOTALL)
_RI_TAG_RE = re.compile(r"</?ri:[^>]*>", re.DOTALL)


def _storage_to_markdown(storage_html: str) -> str:
    """Convert Confluence storage format (XHTML) to markdown.

    Strips Confluence-specific ac: and ri: namespace elements first,
    then converts the remaining HTML to markdown using markdownify.
    """
    try:
        from markdownify import markdownify
    except ImportError:
        raise ImportError(
            "markdownify is required for Confluence support. "
            "Install with: pip install fusesearch[confluence]"
        ) from None

    # Strip ac: and ri: namespace elements
    html = _AC_TAG_RE.sub("", storage_html)
    html = _RI_TAG_RE.sub("", html)

    return markdownify(html, heading_style="ATX", strip=["img"]).strip()


class ConfluenceAdapter(SourceAdapter):
    """Source adapter for Confluence Cloud and Server/Data Center."""

    def __init__(
        self,
        url: str | None = None,
        username: str | None = None,
        api_token: str | None = None,
        password: str | None = None,
        spaces: list[str] | None = None,
        cloud: bool | None = None,
    ):
        self.url = (url or os.getenv("CONFLUENCE_URL", "")).rstrip("/")
        if not self.url:
            raise ValueError(
                "Confluence URL is required. Pass url= or set CONFLUENCE_URL."
            )

        self.username = username or os.getenv("CONFLUENCE_USERNAME", "")
        self.api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN", "")
        self.password = password or os.getenv("CONFLUENCE_PASSWORD", "")

        # Spaces to index (empty = all spaces)
        if spaces is not None:
            self.spaces = spaces
        else:
            env_spaces = os.getenv("CONFLUENCE_SPACES", "")
            self.spaces = [s.strip() for s in env_spaces.split(",") if s.strip()]

        # Auto-detect Cloud vs Server from URL if not explicitly set
        if cloud is not None:
            self.cloud = cloud
        elif os.getenv("CONFLUENCE_CLOUD", "").lower() in ("true", "1"):
            self.cloud = True
        else:
            self.cloud = "atlassian.net" in self.url

        self._client = None

    @property
    def source_type(self) -> str:
        return "confluence"

    def _create_client(self):
        """Create and return the Confluence API client."""
        try:
            from atlassian import Confluence
        except ImportError:
            raise ImportError(
                "atlassian-python-api is required for Confluence support. "
                "Install with: pip install fusesearch[confluence]"
            ) from None

        kwargs = {"url": self.url, "cloud": self.cloud}

        if self.api_token:
            kwargs["username"] = self.username
            kwargs["password"] = self.api_token
        elif self.password:
            kwargs["username"] = self.username
            kwargs["password"] = self.password

        return Confluence(**kwargs)

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _get_spaces(self) -> list[str]:
        """Return configured spaces or discover all spaces from the server."""
        if self.spaces:
            return self.spaces

        logger.info("No spaces configured, discovering all spaces...")
        results = self.client.get_all_spaces(start=0, limit=500, expand=None)
        space_keys = [s["key"] for s in results.get("results", [])]
        logger.info("Discovered %d spaces", len(space_keys))
        return space_keys

    def _page_to_document(self, page: dict, space_key: str) -> Document | None:
        """Convert a Confluence API page response to a Document."""
        body = page.get("body", {}).get("storage", {}).get("value", "")
        if not body:
            return None

        page_id = str(page["id"])
        title = page.get("title", "Untitled")

        content = _storage_to_markdown(body)
        if not content:
            return None

        # Build page URL
        base = self.url
        if self.cloud:
            space_path = f"/wiki/spaces/{space_key}/pages/{page_id}"
        else:
            space_path = f"/pages/viewpage.action?pageId={page_id}"
        url = f"{base}{space_path}"

        # Extract metadata
        version = page.get("version", {})
        ancestors = [
            {"id": str(a["id"]), "title": a.get("title", "")}
            for a in page.get("ancestors", [])
        ]

        return Document(
            source_type=self.source_type,
            source_id=f"confluence:{space_key}:{page_id}",
            title=title,
            content=content,
            url=url,
            metadata={
                "space_key": space_key,
                "page_id": page_id,
                "version": version.get("number", 1),
                "modified_at": version.get("when", ""),
                "ancestors": ancestors,
            },
        )

    def fetch(self) -> Iterator[Document]:
        """Fetch all pages from configured (or all) spaces."""
        for space_key in self._get_spaces():
            logger.info("Fetching pages from space: %s", space_key)
            yield from self._fetch_space(space_key)

    def _fetch_space(self, space_key: str) -> Iterator[Document]:
        """Fetch all pages from a single space with pagination."""
        start = 0
        limit = 50  # Confluence caps body.storage expansion at ~50

        while True:
            pages = self.client.get_all_pages_from_space(
                space_key,
                start=start,
                limit=limit,
                expand="body.storage,version,ancestors",
            )

            if not pages:
                break

            for page in pages:
                doc = self._page_to_document(page, space_key)
                if doc is not None:
                    yield doc

            if len(pages) < limit:
                break
            start += limit

    def fetch_updated(self, since: str | None = None) -> Iterator[Document]:
        """Fetch pages updated since the given ISO datetime.

        Uses CQL lastmodified filter for incremental sync.
        Falls back to full fetch if since is None.
        """
        if since is None:
            yield from self.fetch()
            return

        for space_key in self._get_spaces():
            logger.info(
                "Fetching updated pages from space: %s (since %s)", space_key, since
            )
            yield from self._fetch_space_updated(space_key, since)

    def _fetch_space_updated(self, space_key: str, since: str) -> Iterator[Document]:
        """Fetch pages from a space modified after the given datetime using CQL."""
        # CQL date format: yyyy-MM-dd or yyyy-MM-dd HH:mm
        # Truncate ISO to the format CQL expects
        cql_date = since[:16].replace("T", " ")
        cql = (
            f'space = "{space_key}" AND type = "page" AND lastmodified >= "{cql_date}"'
        )

        start = 0
        limit = 50

        while True:
            results = self.client.cql(
                cql,
                start=start,
                limit=limit,
                expand="body.storage,version,ancestors",
            )
            pages = results.get("results", [])

            if not pages:
                break

            for result in pages:
                # CQL results wrap page in a "content" key for some versions
                page = result.get("content", result)
                doc = self._page_to_document(page, space_key)
                if doc is not None:
                    yield doc

            if len(pages) < limit:
                break
            start += limit
