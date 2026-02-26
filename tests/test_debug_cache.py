import os
from datetime import datetime

from fusesearch.core.debug_cache import dump_documents
from fusesearch.models import Document


def _make_doc(source_type="local_files", source_id="doc-1", title="Test Doc",
              content="# Hello\nWorld", url=None):
    return Document(
        source_type=source_type,
        source_id=source_id,
        title=title,
        content=content,
        url=url,
        fetched_at=datetime(2026, 2, 25, 10, 30, 0),
    )


class TestDumpDocuments:
    def test_creates_folder_per_source_type(self, tmp_path):
        docs = [
            _make_doc(source_type="local_files"),
            _make_doc(source_type="confluence", source_id="conf-1"),
        ]
        dump_documents(docs, str(tmp_path))

        assert (tmp_path / "local_files").is_dir()
        assert (tmp_path / "confluence").is_dir()

    def test_sanitizes_filenames(self, tmp_path):
        doc = _make_doc(source_id="confluence:DEV:12345/page")
        dump_documents([doc], str(tmp_path))

        files = os.listdir(tmp_path / "local_files")
        assert len(files) == 1
        assert files[0] == "confluence_DEV_12345_page.md"

    def test_front_matter_contains_expected_fields(self, tmp_path):
        doc = _make_doc(
            source_id="doc-1",
            title="Setup Guide",
            url="https://example.com/doc",
        )
        dump_documents([doc], str(tmp_path))

        content = (tmp_path / "local_files" / "doc-1.md").read_text()
        assert content.startswith("---\n")
        assert "source_id: doc-1" in content
        assert "title: Setup Guide" in content
        assert "url: https://example.com/doc" in content
        assert "fetched_at: 2026-02-25T10:30:00" in content
        assert content.count("---") == 2

    def test_front_matter_omits_url_when_none(self, tmp_path):
        doc = _make_doc(url=None)
        dump_documents([doc], str(tmp_path))

        content = (tmp_path / "local_files" / "doc-1.md").read_text()
        assert "url:" not in content

    def test_content_preserved_after_front_matter(self, tmp_path):
        doc = _make_doc(content="# Hello\nWorld")
        dump_documents([doc], str(tmp_path))

        content = (tmp_path / "local_files" / "doc-1.md").read_text()
        # Content comes after the closing ---
        parts = content.split("---\n")
        body = parts[2]
        assert body == "# Hello\nWorld"

    def test_clears_directory_on_redump(self, tmp_path):
        docs_v1 = [
            _make_doc(source_id="old-doc", content="old"),
            _make_doc(source_id="keep-doc", content="keep"),
        ]
        dump_documents(docs_v1, str(tmp_path))
        assert (tmp_path / "local_files" / "old-doc.md").exists()

        docs_v2 = [_make_doc(source_id="keep-doc", content="updated")]
        dump_documents(docs_v2, str(tmp_path))

        files = os.listdir(tmp_path / "local_files")
        assert files == ["keep-doc.md"]
        assert not (tmp_path / "local_files" / "old-doc.md").exists()

    def test_returns_file_count(self, tmp_path):
        docs = [
            _make_doc(source_id="a"),
            _make_doc(source_id="b"),
            _make_doc(source_type="confluence", source_id="c"),
        ]
        count = dump_documents(docs, str(tmp_path))
        assert count == 3

    def test_special_characters_in_source_id(self, tmp_path):
        doc = _make_doc(source_id='file:"path*to<doc>|here')
        dump_documents([doc], str(tmp_path))

        files = os.listdir(tmp_path / "local_files")
        assert len(files) == 1
        # All special chars replaced with underscores
        assert ":" not in files[0]
        assert '"' not in files[0]
        assert "*" not in files[0]
        assert "<" not in files[0]
        assert ">" not in files[0]
        assert "|" not in files[0]
