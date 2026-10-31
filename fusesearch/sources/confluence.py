import logging
import os
import re
from collections.abc import Iterator
from datetime import UTC, datetime

from fusesearch.models import Document
from fusesearch.sources.base import SourceAdapter

logger = logging.getLogger(__name__)

_AUTH_HINT = (
    "Check CONFLUENCE_USERNAME and CONFLUENCE_API_TOKEN: API tokens expire, and "
    "scoped tokens only work with the https://api.atlassian.com/ex/confluence/"
    "<cloudId> URL."
)


class ConfluenceError(RuntimeError):
    """A Confluence request failed (credentials, URL, unknown space)."""


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
            self.cloud = "atlassian.net" in self.url or "api.atlassian.com" in self.url

        # Cloud: site root, with or without "/wiki" in the configured URL, e.g.
        # https://myorg.atlassian.net or https://api.atlassian.com/ex/confluence/<id>
        self._site = self.url.removesuffix("/wiki")

        self._client = None

    @property
    def source_type(self) -> str:
        return "confluence"

    # --- HTTP ---

    def _create_client(self):
        """Authenticated Confluence client; only its get(absolute=True) is used, so
        library API changes (4.x -> 5.x moved Cloud to REST v2) don't affect us."""
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

    def _get(self, url: str, params: dict | None = None) -> dict:
        """GET a REST URL and return its JSON, or raise ConfluenceError."""
        import requests

        try:
            data = self.client.get(url, params=params, absolute=True)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            # Unauthenticated calls get 403 (v1) or 404 (v2) on Cloud.
            hint = f" {_AUTH_HINT}" if status in (401, 403, 404) else ""
            raise ConfluenceError(
                f"Confluence request failed ({status}): {_path(url)}: {e}.{hint}"
            ) from None
        if not isinstance(data, dict):
            # e.g. a redirect to the login page instead of JSON
            raise ConfluenceError(
                f"Confluence returned no JSON for {_path(url)}. {_AUTH_HINT}"
            )
        return data

    def _paged(
        self, url: str, params: dict, cloud: bool
    ) -> Iterator[tuple[dict, dict]]:
        """Yield (item, response _links) for every result, following _links.next.

        Cloud v2 next links are relative to the site root (they start with /wiki);
        v1 next links are relative to _links.base (which includes the context path).
        """
        while url:
            data = self._get(url, params)
            links = data.get("_links", {})
            for item in data.get("results", []):
                yield item, links
            nxt = links.get("next")
            if not nxt:
                break
            base = self._site if cloud else links.get("base", self.url)
            url, params = f"{base}{nxt}", None

    # --- Fetching ---

    def fetch(self) -> Iterator[Document]:
        """Fetch all pages from configured (or all) spaces."""
        if self.cloud:
            yield from self._cloud_fetch(since=None)
        else:
            for space_key in self._server_space_keys():
                logger.info("Fetching pages from space: %s", space_key)
                yield from self._server_fetch_space(space_key)

    def fetch_updated(self, since: str | None = None) -> Iterator[Document]:
        """Fetch pages updated since the given ISO datetime (full fetch if None).

        Cloud (REST v2 has no "modified since" filter): pages are listed and
        filtered on version.createdAt. Server/Data Center: CQL lastmodified.
        """
        if since is None:
            yield from self.fetch()
        elif self.cloud:
            yield from self._cloud_fetch(since=_parse_time(since))
        else:
            for space_key in self._server_space_keys():
                logger.info("Fetching pages from %s updated since %s", space_key, since)
                yield from self._server_fetch_space_updated(space_key, since)

    # --- Cloud: REST v2 ---

    def _v2(self, path: str) -> str:
        return f"{self._site}/wiki/api/v2/{path}"

    def _cloud_spaces(self) -> list[dict]:
        params = {"limit": 250}
        if self.spaces:
            params["keys"] = ",".join(self.spaces)
        else:
            logger.info("No spaces configured, discovering all spaces...")
        spaces = [s for s, _ in self._paged(self._v2("spaces"), params, cloud=True)]
        if self.spaces:
            # A typo must not look like an empty space: its pages would be deleted.
            found = {s["key"] for s in spaces}
            missing = [k for k in self.spaces if k not in found]
            if missing:
                raise ConfluenceError(
                    f"Confluence space(s) not found or not visible: {', '.join(missing)}"
                )
        return spaces

    def _cloud_fetch(self, since) -> Iterator[Document]:
        for space in self._cloud_spaces():
            key = space["key"]
            logger.info("Fetching pages from space: %s", key)
            url = self._v2(f"spaces/{space['id']}/pages")
            params = {"body-format": "storage", "limit": 250}
            pages = list(self._paged(url, params, cloud=True))
            by_id = {str(p["id"]): p for p, _ in pages}
            for page, links in pages:
                if since is not None:
                    modified = page.get("version", {}).get("createdAt")
                    if not modified or _parse_time(modified) < since:
                        continue
                doc = self._v2_page_to_document(page, key, by_id, links.get("base"))
                if doc is not None:
                    yield doc

    def _v2_page_to_document(
        self, page: dict, space_key: str, by_id: dict, links_base: str | None
    ) -> Document | None:
        page_id = str(page["id"])
        webui = page.get("_links", {}).get("webui")
        if links_base and webui:
            url = f"{links_base}{webui}"
        else:
            url = f"{self._site}/wiki/spaces/{space_key}/pages/{page_id}"
        version = page.get("version", {})
        return self._make_document(
            space_key=space_key,
            page_id=page_id,
            title=page.get("title", "Untitled"),
            storage=page.get("body", {}).get("storage", {}).get("value", ""),
            url=url,
            version=version.get("number", 1),
            modified_at=version.get("createdAt", ""),
            ancestors=_v2_ancestors(page, by_id),
        )

    # --- Server / Data Center: REST v1 ---

    def _v1(self, path: str) -> str:
        return f"{self.url}/rest/api/{path}"

    def _server_space_keys(self) -> list[str]:
        if self.spaces:
            return self.spaces
        logger.info("No spaces configured, discovering all spaces...")
        spaces = self._paged(self._v1("space"), {"limit": 500}, cloud=False)
        keys = [s["key"] for s, _ in spaces]
        logger.info("Discovered %d spaces", len(keys))
        return keys

    def _server_fetch_space(self, space_key: str) -> Iterator[Document]:
        params = {
            "spaceKey": space_key,
            "type": "page",
            "expand": _V1_EXPAND,
            "limit": 50,  # Confluence caps body.storage expansion at ~50
        }
        for page, _ in self._paged(self._v1("content"), params, cloud=False):
            doc = self._page_to_document(page, space_key)
            if doc is not None:
                yield doc

    def _server_fetch_space_updated(
        self, space_key: str, since: str
    ) -> Iterator[Document]:
        # CQL date format: yyyy-MM-dd HH:mm
        cql_date = since[:16].replace("T", " ")
        params = {
            "cql": f'space = "{space_key}" AND type = "page" '
            f'AND lastmodified >= "{cql_date}"',
            "expand": _V1_EXPAND,
            "limit": 50,
        }
        for result, _ in self._paged(self._v1("content/search"), params, cloud=False):
            page = result.get("content", result)  # /search wraps it, content/search not
            doc = self._page_to_document(page, space_key)
            if doc is not None:
                yield doc

    def _page_to_document(self, page: dict, space_key: str) -> Document | None:
        """Convert a REST v1 page (Server/Data Center) to a Document."""
        page_id = str(page["id"])
        if self.cloud:
            url = f"{self._site}/wiki/spaces/{space_key}/pages/{page_id}"
        else:
            url = f"{self.url}/pages/viewpage.action?pageId={page_id}"
        version = page.get("version", {})
        return self._make_document(
            space_key=space_key,
            page_id=page_id,
            title=page.get("title", "Untitled"),
            storage=page.get("body", {}).get("storage", {}).get("value", ""),
            url=url,
            version=version.get("number", 1),
            modified_at=version.get("when", ""),
            ancestors=[
                {"id": str(a["id"]), "title": a.get("title", "")}
                for a in page.get("ancestors", [])
            ],
        )

    def _make_document(
        self,
        *,
        space_key,
        page_id,
        title,
        storage,
        url,
        version,
        modified_at,
        ancestors,
    ) -> Document | None:
        if not storage:
            return None
        content = _storage_to_markdown(storage)
        if not content:
            return None
        return Document(
            source_type=self.source_type,
            source_id=f"confluence:{space_key}:{page_id}",
            title=title,
            content=content,
            url=url,
            metadata={
                "space_key": space_key,
                "page_id": page_id,
                "version": version,
                "modified_at": modified_at,
                "ancestors": ancestors,
            },
        )


_V1_EXPAND = "body.storage,version,ancestors"


def _path(url: str) -> str:
    """URL without scheme and host, for error messages."""
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    return parts.path + (f"?{parts.query}" if parts.query else "")


def _parse_time(value: str) -> datetime:
    """ISO datetime (``Z`` allowed); naive values are taken as UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _v2_ancestors(page: dict, by_id: dict) -> list[dict]:
    """Root-first ancestor chain from parentId links between the space's pages
    (v2 page lists carry parentId but not the full ancestor list)."""
    chain = []
    seen = set()
    parent_id = page.get("parentId")
    while parent_id and str(parent_id) in by_id and str(parent_id) not in seen:
        seen.add(str(parent_id))
        parent = by_id[str(parent_id)]
        chain.append({"id": str(parent_id), "title": parent.get("title", "")})
        parent_id = parent.get("parentId")
    return list(reversed(chain))
