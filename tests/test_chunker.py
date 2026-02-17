from fusesearch.core.chunker import chunk_document, _split_by_headings, _split_by_size
from fusesearch.models import Document


def _make_doc(content, **kwargs):
    defaults = dict(source_type="test", source_id="doc-1", title="Test Doc")
    defaults.update(kwargs)
    return Document(content=content, **defaults)


# --- _split_by_headings ---


class TestSplitByHeadings:
    def test_no_headings(self):
        sections = _split_by_headings("Just plain text.\n\nAnother paragraph.")
        assert len(sections) == 1
        assert sections[0][0] == []
        assert "Just plain text." in sections[0][1]

    def test_single_heading(self):
        text = "# Title\n\nBody text here."
        sections = _split_by_headings(text)
        assert len(sections) == 1
        assert sections[0][0] == ["Title"]
        assert sections[0][1] == "Body text here."

    def test_preamble_before_first_heading(self):
        text = "Preamble text.\n\n# First Heading\n\nBody."
        sections = _split_by_headings(text)
        assert len(sections) == 2
        assert sections[0][0] == []
        assert sections[0][1] == "Preamble text."
        assert sections[1][0] == ["First Heading"]

    def test_nested_headings_build_path(self):
        text = "# H1\n\nIntro\n\n## H2\n\nDetails\n\n### H3\n\nDeep details"
        sections = _split_by_headings(text)
        paths = [s[0] for s in sections]
        assert ["H1"] in paths
        assert ["H1", "H2"] in paths
        assert ["H1", "H2", "H3"] in paths

    def test_heading_level_reset(self):
        text = "# A\n\nA content\n\n## A1\n\nA1 content\n\n# B\n\nB content"
        sections = _split_by_headings(text)
        # B should reset the stack, not be under A
        b_section = [s for s in sections if "B content" in s[1]][0]
        assert b_section[0] == ["B"]

    def test_empty_section_skipped(self):
        text = "# Empty\n\n# Has Content\n\nSome text"
        sections = _split_by_headings(text)
        # The empty section between two headings should be skipped
        assert all(s[1].strip() for s in sections)


# --- _split_by_size ---


class TestSplitBySize:
    def test_small_text_returns_single(self):
        pieces = _split_by_size("short text", max_size=100)
        assert pieces == ["short text"]

    def test_splits_on_paragraph_boundary(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        pieces = _split_by_size(text, max_size=30)
        assert len(pieces) > 1
        # Each piece should be within limit
        for p in pieces:
            assert len(p) <= 30

    def test_hard_splits_oversized_paragraph(self):
        text = "x" * 250
        pieces = _split_by_size(text, max_size=100)
        assert len(pieces) == 3
        assert pieces[0] == "x" * 100
        assert pieces[1] == "x" * 100
        assert pieces[2] == "x" * 50

    def test_accumulates_small_paragraphs(self):
        text = "a\n\nb\n\nc"
        pieces = _split_by_size(text, max_size=100)
        assert len(pieces) == 1
        assert pieces[0] == "a\n\nb\n\nc"


# --- chunk_document ---


class TestChunkDocument:
    def test_basic_chunking(self):
        doc = _make_doc("# Title\n\nSome content here.")
        chunks = chunk_document(doc)
        assert len(chunks) >= 1
        assert chunks[0].title == "Test Doc"
        assert chunks[0].source_type == "test"
        assert chunks[0].document_source_id == "doc-1"

    def test_heading_path_preserved(self):
        doc = _make_doc("# Top\n\n## Sub\n\nContent under sub.")
        chunks = chunk_document(doc)
        sub_chunk = [c for c in chunks if "Content under sub" in c.content][0]
        assert sub_chunk.heading_path == ["Top", "Sub"]

    def test_chunk_index_increments(self):
        doc = _make_doc("# A\n\nContent A\n\n# B\n\nContent B\n\n# C\n\nContent C")
        chunks = chunk_document(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_content_produces_no_chunks(self):
        doc = _make_doc("")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_whitespace_only_produces_no_chunks(self):
        doc = _make_doc("   \n\n   ")
        chunks = chunk_document(doc)
        assert chunks == []

    def test_max_chunk_size_respected(self):
        long_section = "word " * 500  # ~2500 chars
        doc = _make_doc(f"# Section\n\n{long_section}")
        chunks = chunk_document(doc, max_chunk_size=200)
        for chunk in chunks:
            assert len(chunk.content) <= 200

    def test_metadata_propagated(self):
        doc = _make_doc("# Test\n\nContent.", metadata={"key": "value"})
        chunks = chunk_document(doc)
        assert chunks[0].metadata == {"key": "value"}

    def test_url_propagated(self):
        doc = _make_doc("# Test\n\nContent.", url="https://example.com")
        chunks = chunk_document(doc)
        assert chunks[0].url == "https://example.com"

    def test_content_hash_is_set(self):
        doc = _make_doc("# Test\n\nContent.")
        chunks = chunk_document(doc)
        assert chunks[0].content_hash
        assert len(chunks[0].content_hash) == 64  # SHA-256 hex
