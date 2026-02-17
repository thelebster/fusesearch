from importlib.metadata import version

from fusesearch.core.embedder import Embedder, create_embedder
from fusesearch.core.reranker import Reranker, create_reranker
from fusesearch.core.synthesizer import synthesize
from fusesearch.indexer import Indexer
from fusesearch.llm import create_llm
from fusesearch.models import Chunk, Document
from fusesearch.sources.base import SourceAdapter
from fusesearch.sources.local_files import LocalFilesAdapter
from fusesearch.store.qdrant import QdrantStore

__version__ = version("fusesearch")

__all__ = [
    "Chunk",
    "Document",
    "Embedder",
    "Indexer",
    "LocalFilesAdapter",
    "QdrantStore",
    "Reranker",
    "SourceAdapter",
    "create_embedder",
    "create_llm",
    "create_reranker",
    "synthesize",
    "__version__",
]
