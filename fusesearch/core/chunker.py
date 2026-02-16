import re

from fusesearch.models import Chunk, Document

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_document(document: Document, max_chunk_size: int = 1000) -> list[Chunk]:
    """Split a document into chunks by markdown headings.

    Splits on headings first (semantic boundaries), then splits
    oversized sections by paragraphs.
    """
    sections = _split_by_headings(document.content)
    chunks: list[Chunk] = []

    for heading_path, content in sections:
        for piece in _split_by_size(content, max_chunk_size):
            chunk = Chunk(
                document_source_id=document.source_id,
                source_type=document.source_type,
                title=document.title,
                content=piece.strip(),
                url=document.url,
                metadata=document.metadata,
                heading_path=heading_path,
                chunk_index=len(chunks),
            )
            if chunk.content:
                chunks.append(chunk)

    return chunks


def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
    """Split markdown text by headings, tracking the heading hierarchy."""
    matches = list(HEADING_PATTERN.finditer(text))

    if not matches:
        return [([], text)]

    sections: list[tuple[list[str], str]] = []
    heading_stack: list[tuple[int, str]] = []

    # Content before the first heading
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(([], preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()

        # Update heading stack — pop headings at same or deeper level
        heading_stack = [(l, t) for l, t in heading_stack if l < level]
        heading_stack.append((level, title))

        # Extract content between this heading and the next
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if content:
            path = [t for _, t in heading_stack]
            sections.append((path, content))

    return sections


def _split_by_size(text: str, max_size: int) -> list[str]:
    """Split text into pieces that fit within max_size, splitting on paragraphs."""
    if len(text) <= max_size:
        return [text]

    paragraphs = text.split("\n\n")
    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > max_size:
            pieces.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        pieces.append(current)

    return pieces
