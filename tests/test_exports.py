def test_all_public_names_importable():
    from fusesearch import (
        Chunk,
        Document,
        Embedder,
        Indexer,
        LocalFilesAdapter,
        QdrantStore,
        Reranker,
        SourceAdapter,
        create_adapter,
        create_embedder,
        create_llm,
        create_reranker,
        synthesize,
    )

    # Verify they are the expected types
    assert callable(create_adapter)
    assert callable(create_embedder)
    assert callable(create_llm)
    assert callable(create_reranker)
    assert callable(synthesize)

    # Verify classes are actual classes
    for cls in (
        Document,
        Chunk,
        Indexer,
        QdrantStore,
        Embedder,
        Reranker,
        SourceAdapter,
        LocalFilesAdapter,
    ):
        assert isinstance(cls, type)


def test_version_is_string():
    from fusesearch import __version__

    assert isinstance(__version__, str)
    assert len(__version__) > 0
