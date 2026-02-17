from fusesearch.core.synthesizer import build_context


def _result(title="Doc", content="Some text", heading_path=None):
    return {
        "title": title,
        "content": content,
        "heading_path": heading_path or [],
    }


class TestBuildContext:
    def test_single_result(self):
        ctx = build_context([_result(title="My Doc", content="Hello world")])
        assert "[1] My Doc" in ctx
        assert "Hello world" in ctx

    def test_multiple_results_numbered(self):
        results = [
            _result(title="First", content="AAA"),
            _result(title="Second", content="BBB"),
            _result(title="Third", content="CCC"),
        ]
        ctx = build_context(results)
        assert "[1] First" in ctx
        assert "[2] Second" in ctx
        assert "[3] Third" in ctx

    def test_heading_path_included(self):
        result = _result(
            title="Doc",
            content="Text",
            heading_path=["Chapter 1", "Section A"],
        )
        ctx = build_context([result])
        assert "Chapter 1" in ctx
        assert "Section A" in ctx
        assert " > " in ctx

    def test_no_heading_path(self):
        ctx = build_context([_result(title="Doc", heading_path=[])])
        assert " > " not in ctx

    def test_results_separated(self):
        results = [_result(content="AAA"), _result(content="BBB")]
        ctx = build_context(results)
        assert "---" in ctx

    def test_empty_results(self):
        ctx = build_context([])
        assert ctx == ""

    def test_missing_fields_use_defaults(self):
        result = {"content": "text"}  # no title, no heading_path
        ctx = build_context([result])
        assert "[1] Untitled" in ctx
        assert "text" in ctx
