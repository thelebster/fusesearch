from hashlib import sha256

from fusesearch.models import Chunk


class TestChunkContentHash:
    def test_deterministic(self):
        chunk1 = Chunk(
            document_source_id="doc-1",
            source_type="confluence",
            title="Setup Guide",
            content="Same content.",
        )
        chunk2 = Chunk(
            document_source_id="doc-2",
            source_type="local_files",
            title="Install Notes",
            content="Same content.",
        )
        assert chunk1.content_hash == chunk2.content_hash

    def test_matches_sha256(self):
        content = "Install Docker and run make start."
        chunk = Chunk(
            document_source_id="doc-1",
            source_type="local_files",
            title="Setup Guide",
            content=content,
        )
        expected = sha256(content.encode()).hexdigest()
        assert chunk.content_hash == expected

    def test_different_content_different_hash(self):
        chunk1 = Chunk(
            document_source_id="doc-1",
            source_type="local_files",
            title="Setup Guide",
            content="Install Docker and run make start.",
        )
        chunk2 = Chunk(
            document_source_id="doc-1",
            source_type="local_files",
            title="Setup Guide",
            content="Configure environment variables in .env file.",
        )
        assert chunk1.content_hash != chunk2.content_hash
