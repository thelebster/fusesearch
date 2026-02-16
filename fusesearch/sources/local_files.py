from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from fusesearch.models import Document
from fusesearch.sources.base import SourceAdapter

SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst"}


class LocalFilesAdapter(SourceAdapter):
    """Source adapter for local markdown and text files."""

    def __init__(self, directories: list[str | Path]):
        self.directories = [Path(d) for d in directories]

    @property
    def source_type(self) -> str:
        return "local_files"

    def fetch(self) -> Iterator[Document]:
        for directory in self.directories:
            yield from self._scan_directory(directory)

    def fetch_updated(self, since: str | None = None) -> Iterator[Document]:
        if since is None:
            yield from self.fetch()
            return

        cutoff = datetime.fromisoformat(since)
        for document in self.fetch():
            if document.fetched_at >= cutoff:
                yield document

    def _scan_directory(self, directory: Path) -> Iterator[Document]:
        if not directory.is_dir():
            return

        for path in directory.rglob("*"):
            if path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            if not path.is_file():
                continue

            content = path.read_text(encoding="utf-8", errors="replace")
            stat = path.stat()

            yield Document(
                source_type=self.source_type,
                source_id=str(path.resolve()),
                title=path.stem,
                content=content,
                metadata={
                    "path": str(path.resolve()),
                    "extension": path.suffix,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                },
            )
