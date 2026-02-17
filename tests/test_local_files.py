import tempfile
from pathlib import Path

from fusesearch.sources.local_files import LocalFilesAdapter


def _write_file(directory, name, content="Sample content."):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestLocalFilesAdapter:
    def test_reads_markdown_files(self):
        with tempfile.TemporaryDirectory() as d:
            _write_file(d, "doc.md", "# Hello\n\nWorld")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch())
            assert len(docs) == 1
            assert docs[0].title == "doc"
            assert "Hello" in docs[0].content
            assert docs[0].source_type == "local_files"

    def test_reads_txt_and_rst(self):
        with tempfile.TemporaryDirectory() as d:
            _write_file(d, "readme.txt", "Plain text.")
            _write_file(d, "guide.rst", "RST content.")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch())
            assert len(docs) == 2
            titles = {doc.title for doc in docs}
            assert titles == {"readme", "guide"}

    def test_ignores_unsupported_extensions(self):
        with tempfile.TemporaryDirectory() as d:
            _write_file(d, "data.json", '{"key": "value"}')
            _write_file(d, "image.png", "binary data")
            _write_file(d, "doc.md", "Valid markdown.")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch())
            assert len(docs) == 1
            assert docs[0].title == "doc"

    def test_scans_nested_directories(self):
        with tempfile.TemporaryDirectory() as d:
            _write_file(d, "top.md", "Top level.")
            _write_file(d, "sub/nested.md", "Nested file.")
            _write_file(d, "sub/deep/deep.md", "Deep file.")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch())
            assert len(docs) == 3

    def test_multiple_directories(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _write_file(d1, "a.md", "From dir 1.")
            _write_file(d2, "b.md", "From dir 2.")
            adapter = LocalFilesAdapter(directories=[d1, d2])
            docs = list(adapter.fetch())
            assert len(docs) == 2

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as d:
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch())
            assert docs == []

    def test_nonexistent_directory(self):
        adapter = LocalFilesAdapter(directories=["/nonexistent/path"])
        docs = list(adapter.fetch())
        assert docs == []

    def test_metadata_includes_path_and_extension(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write_file(d, "doc.md", "Content.")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch())
            meta = docs[0].metadata
            assert meta["extension"] == ".md"
            assert meta["path"] == str(path.resolve())
            assert "size_bytes" in meta
            assert "modified_at" in meta

    def test_source_id_is_absolute_path(self):
        with tempfile.TemporaryDirectory() as d:
            _write_file(d, "doc.md", "Content.")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch())
            assert docs[0].source_id.startswith("/")

    def test_fetch_updated_without_since_returns_all(self):
        with tempfile.TemporaryDirectory() as d:
            _write_file(d, "a.md", "A")
            _write_file(d, "b.md", "B")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch_updated(since=None))
            assert len(docs) == 2

    def test_fetch_updated_with_future_since_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            _write_file(d, "a.md", "Old content")
            adapter = LocalFilesAdapter(directories=[d])
            docs = list(adapter.fetch_updated(since="2099-01-01T00:00:00"))
            assert docs == []
