from abc import ABC, abstractmethod
from collections.abc import Iterator

from fusesearch.models import Document


class SourceAdapter(ABC):
    """Abstract base class for all source adapters."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Unique identifier for this source type (e.g. 'local_files')."""

    @abstractmethod
    def fetch(self) -> Iterator[Document]:
        """Fetch all documents from this source."""

    @abstractmethod
    def fetch_updated(self, since: str | None = None) -> Iterator[Document]:
        """Fetch only documents updated since the given cursor.

        The cursor format is source-specific (e.g. timestamp, page token).
        If None, behaves like fetch() (full sync).
        """
