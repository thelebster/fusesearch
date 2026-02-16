import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fusesearch import __version__
from fusesearch.core.embedder import create_embedder
from fusesearch.core.reranker import rerank_enabled
from fusesearch.indexer import Indexer
from fusesearch.sources.local_files import LocalFilesAdapter
from fusesearch.store.qdrant import QdrantStore

app = FastAPI(title="FuseSearch", version=__version__)

embedder = create_embedder()

qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
store = QdrantStore(
    host=qdrant_host,
    port=qdrant_port,
    dimension=embedder.dimension,
    collection_suffix=embedder.name,
)
indexer = Indexer(store=store, embedder=embedder)

# Lazy-initialized reranker (only loaded when first rerank request arrives)
_reranker = None

RERANK_OVERFETCH = 3


def _get_reranker():
    global _reranker
    if _reranker is None:
        from fusesearch.core.reranker import create_reranker

        _reranker = create_reranker()
    return _reranker


class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    hybrid: bool = os.getenv("FUSESEARCH_HYBRID", "true").lower() == "true"
    rerank: bool = rerank_enabled()
    vector_weight: float = 0.7


class AskRequest(BaseModel):
    query: str
    limit: int = 5
    hybrid: bool = os.getenv("FUSESEARCH_HYBRID", "true").lower() == "true"
    rerank: bool = rerank_enabled()
    vector_weight: float = 0.7


class IndexRequest(BaseModel):
    paths: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": store.count()}


@app.post("/search")
def search(req: SearchRequest):
    fetch_limit = req.limit * RERANK_OVERFETCH if req.rerank else req.limit
    query_vector = embedder.embed_one(req.query)
    if req.hybrid:
        results = store.hybrid_search(
            query_vector, req.query, limit=fetch_limit, vector_weight=req.vector_weight
        )
    else:
        results = store.search(query_vector, limit=fetch_limit)
    if req.rerank:
        reranker = _get_reranker()
        results = reranker.rerank(req.query, results, limit=req.limit)
    return {"results": results}


@app.post("/ask")
def ask(req: AskRequest):
    from fusesearch.core.synthesizer import synthesize
    from fusesearch.llm import create_llm

    fetch_limit = req.limit * RERANK_OVERFETCH if req.rerank else req.limit
    query_vector = embedder.embed_one(req.query)
    if req.hybrid:
        results = store.hybrid_search(
            query_vector, req.query, limit=fetch_limit, vector_weight=req.vector_weight
        )
    else:
        results = store.search(query_vector, limit=fetch_limit)
    if req.rerank:
        reranker = _get_reranker()
        results = reranker.rerank(req.query, results, limit=req.limit)

    try:
        llm = create_llm()
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None

    return synthesize(llm, req.query, results)


@app.post("/index")
def index(req: IndexRequest):
    adapter = LocalFilesAdapter(directories=req.paths)
    documents = list(adapter.fetch())
    stats = indexer.index_documents(documents)
    return {"documents_found": len(documents), **stats}
