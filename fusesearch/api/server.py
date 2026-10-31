from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fusesearch import __version__
from fusesearch.config import load_settings
from fusesearch.core.embedder import create_embedder
from fusesearch.core.reranker import rerank_enabled
from fusesearch.indexer import Indexer
from fusesearch.sources import create_adapter
from fusesearch.store.qdrant import QdrantStore

settings = load_settings()

app = FastAPI(title="FuseSearch", version=__version__)

embedder = create_embedder(settings.embedder_provider)

store = QdrantStore(
    host=settings.qdrant_host,
    port=settings.qdrant_port,
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
    hybrid: bool = settings.hybrid
    rerank: bool = rerank_enabled()
    vector_weight: float = 0.7


class AskRequest(BaseModel):
    query: str
    limit: int = 5
    hybrid: bool = settings.hybrid
    rerank: bool = rerank_enabled()
    vector_weight: float = 0.7


class IndexRequest(BaseModel):
    source: str = "local_files"
    source_config: dict = {}
    since: str | None = None
    debug_cache: bool = settings.debug_cache_enabled
    cache_dir: str = settings.debug_cache_dir


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
    # Support named source from config file
    if req.source in settings.sources and not req.source_config:
        src = settings.sources[req.source]
        source_config = src.model_dump(exclude_none=True)
        adapter = create_adapter(src.type, source_config)
    else:
        adapter = create_adapter(req.source, req.source_config)

    if req.since:
        documents = list(adapter.fetch_updated(since=req.since))
    else:
        documents = list(adapter.fetch())
    cache_dir = req.cache_dir if req.debug_cache else None
    idx = Indexer(store=store, embedder=embedder, cache_dir=cache_dir)
    stats = idx.index_documents(documents)
    return {"documents_found": len(documents), **stats}
