from unittest.mock import MagicMock

import pytest

from fusesearch.indexer import Indexer
from fusesearch.models import Document
from fusesearch.store.qdrant import QdrantStore, point_id
from tests.fakes import FakeEmbedder, FakeStore


def _doc(source_id, content, source_type="local_files"):
    return Document(
        source_type=source_type, source_id=source_id, title=source_id, content=content
    )


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def indexer(store):
    return Indexer(store=store, embedder=FakeEmbedder())


def test_indexing_second_source_keeps_first(indexer, store):
    indexer.index_documents([_doc("a.md", "Alpha text")], source="blog")
    stats = indexer.index_documents([_doc("b.md", "Beta text")], source="wiki")

    assert stats["deleted"] == 0
    assert store.sources() == ["blog", "wiki"]


def test_rerun_unchanged_skips_all_and_deletes_nothing(indexer):
    docs = [_doc("a.md", "Alpha text"), _doc("b.md", "Beta text")]
    indexer.index_documents(docs, source="blog")
    stats = indexer.index_documents(docs, source="blog")

    assert (stats["new"], stats["skipped"], stats["deleted"]) == (0, 2, 0)


def test_removed_document_deleted_only_within_its_source(indexer, store):
    indexer.index_documents([_doc("w.md", "Wiki text")], source="wiki")
    indexer.index_documents(
        [_doc("a.md", "Alpha text"), _doc("b.md", "Beta text")], source="blog"
    )
    stats = indexer.index_documents([_doc("a.md", "Alpha text")], source="blog")

    assert stats["deleted"] == 1
    assert store.contents("blog") == ["Alpha text"]
    assert store.contents("wiki") == ["Wiki text"]


def test_same_text_in_two_sources_is_stored_twice(indexer, store):
    indexer.index_documents([_doc("a.md", "Shared text")], source="blog")
    indexer.index_documents([_doc("x.md", "Shared text")], source="wiki")
    assert len(store.points) == 2
    indexer.index_documents([_doc("other.md", "Other")], source="wiki")

    # Removing the shared text from wiki must not remove it from blog.
    assert store.contents("blog") == ["Shared text"]
    assert store.contents("wiki") == ["Other"]


def test_incremental_run_never_deletes(indexer, store):
    indexer.index_documents(
        [_doc("a.md", "Alpha text"), _doc("b.md", "Beta text")], source="blog"
    )
    stats = indexer.index_documents(
        [_doc("c.md", "Gamma text")], source="blog", incremental=True
    )

    assert (stats["new"], stats["deleted"]) == (1, 0)
    assert store.contents("blog") == ["Alpha text", "Beta text", "Gamma text"]


def test_empty_full_sync_keeps_existing_chunks(indexer, store, capsys):
    indexer.index_documents([_doc("a.md", "Alpha text")], source="wiki")
    stats = indexer.index_documents([], source="wiki")

    assert stats["deleted"] == 0
    assert store.contents("wiki") == ["Alpha text"]
    assert "0 documents fetched" in capsys.readouterr().out


def test_source_defaults_to_source_type(indexer, store):
    stats = indexer.index_documents([_doc("a.md", "Alpha text")])

    assert stats["source"] == "local_files"
    assert store.sources() == ["local_files"]


def test_mixed_source_types_without_source_raise(indexer):
    docs = [_doc("a.md", "A"), _doc("p1", "B", source_type="confluence")]
    with pytest.raises(ValueError, match="pass source="):
        indexer.index_documents(docs)


def test_no_documents_and_no_source_is_a_noop(indexer, store):
    stats = indexer.index_documents([])

    assert (stats["new"], stats["deleted"]) == (0, 0)
    assert store.points == {}


# --- QdrantStore with a mocked client (no Qdrant needed) ---


def _store_with_mock_client():
    store = QdrantStore.__new__(QdrantStore)
    store.client = MagicMock()
    store.collection_name = "test"
    return store


def test_point_id_differs_per_source():
    assert point_id("blog", "ab" * 32) != point_id("wiki", "ab" * 32)
    assert point_id("blog", "ab" * 32) == point_id("blog", "ab" * 32)


def test_upsert_writes_source_payload_and_scoped_id():
    from fusesearch.core.chunker import chunk_document

    store = _store_with_mock_client()
    chunk = chunk_document(_doc("a.md", "Alpha text"))[0]
    chunk.source = "blog"
    store.upsert([chunk], [[0.1, 0.2, 0.3]])

    point = store.client.upsert.call_args.kwargs["points"][0]
    assert point.payload["source"] == "blog"
    assert point.id == point_id("blog", chunk.content_hash)


def test_get_existing_hashes_filters_by_source():
    store = _store_with_mock_client()
    store.client.scroll.return_value = ([], None)

    store.get_existing_hashes(source="blog")
    flt = store.client.scroll.call_args.kwargs["scroll_filter"]
    assert flt.must[0].key == "source"
    assert flt.must[0].match.value == "blog"

    store.get_existing_hashes()
    assert store.client.scroll.call_args.kwargs["scroll_filter"] is None
