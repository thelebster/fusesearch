import os

from fastapi import FastAPI
from pydantic import BaseModel

from fusesearch import __version__
from fusesearch.core.embedder import LocalEmbedder
from fusesearch.indexer import Indexer
from fusesearch.sources.local_files import LocalFilesAdapter
from fusesearch.store.qdrant import QdrantStore

app = FastAPI(title="FuseSearch", version=__version__)

embedder = LocalEmbedder()

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
store = QdrantStore(host=qdrant_host, port=qdrant_port, dimension=embedder.dimension)
indexer = Indexer(store=store, embedder=embedder)


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    hybrid: bool = True
    vector_weight: float = 0.7


class IndexRequest(BaseModel):
    paths: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": store.count()}


@app.post("/search")
def search(req: SearchRequest):
    query_vector = embedder.embed_one(req.query)
    if req.hybrid:
        results = store.hybrid_search(
            query_vector, req.query, limit=req.limit, vector_weight=req.vector_weight
        )
    else:
        results = store.search(query_vector, limit=req.limit)
    return {"results": results}


@app.post("/index")
def index(req: IndexRequest):
    adapter = LocalFilesAdapter(directories=req.paths)
    documents = list(adapter.fetch())
    stats = indexer.index_documents(documents)
    return {"documents_found": len(documents), **stats}
