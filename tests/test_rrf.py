from fusesearch.store.qdrant import QdrantStore


def _result(id, score=0.0, content="text"):
    return {
        "_id": id,
        "score": score,
        "content": content,
        "title": f"Doc {id}",
        "source_type": "test",
        "heading_path": [],
        "metadata": {},
    }


class TestRRFFuse:
    def test_empty_inputs(self):
        result = QdrantStore._rrf_fuse([], [])
        assert result == []

    def test_vector_only(self):
        vector = [_result("a"), _result("b")]
        result = QdrantStore._rrf_fuse(vector, [])
        assert len(result) == 2
        assert result[0]["_id"] == "a"  # rank 1 > rank 2

    def test_keyword_only(self):
        keyword = [_result("x"), _result("y")]
        result = QdrantStore._rrf_fuse([], keyword)
        assert len(result) == 2
        assert result[0]["_id"] == "x"

    def test_deduplication(self):
        """Same doc in both lists should appear once with combined score."""
        vector = [_result("shared"), _result("vec-only")]
        keyword = [_result("shared"), _result("kw-only")]
        result = QdrantStore._rrf_fuse(vector, keyword)
        ids = [r["_id"] for r in result]
        assert ids.count("shared") == 1
        assert len(result) == 3

    def test_shared_doc_ranks_higher(self):
        """A doc appearing in both lists should rank above docs in only one."""
        vector = [_result("shared"), _result("vec-only")]
        keyword = [_result("shared"), _result("kw-only")]
        result = QdrantStore._rrf_fuse(vector, keyword)
        assert result[0]["_id"] == "shared"

    def test_vector_weight(self):
        """Higher vector weight should favor vector results."""
        vector = [_result("vec-top"), _result("shared")]
        keyword = [_result("kw-top"), _result("shared")]

        # Heavy vector weight
        vec_heavy = QdrantStore._rrf_fuse(vector, keyword, vector_weight=0.9)
        # Heavy keyword weight
        kw_heavy = QdrantStore._rrf_fuse(vector, keyword, vector_weight=0.1)

        # shared is always top (appears in both), but second place differs
        vec_heavy_ids = [r["_id"] for r in vec_heavy]
        kw_heavy_ids = [r["_id"] for r in kw_heavy]

        # After "shared", vec-top should be #2 with heavy vector weight
        assert vec_heavy_ids[1] == "vec-top"
        # After "shared", kw-top should be #2 with heavy keyword weight
        assert kw_heavy_ids[1] == "kw-top"

    def test_scores_are_positive(self):
        vector = [_result("a"), _result("b")]
        keyword = [_result("c")]
        result = QdrantStore._rrf_fuse(vector, keyword)
        for r in result:
            assert r["score"] > 0

    def test_results_sorted_descending(self):
        vector = [_result("a"), _result("b"), _result("c")]
        keyword = [_result("b"), _result("d")]
        result = QdrantStore._rrf_fuse(vector, keyword)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)
