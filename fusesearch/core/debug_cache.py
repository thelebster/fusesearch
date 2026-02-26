"""Debug cache: dump pre-chunking documents to disk for inspection."""

import os
import re

from fusesearch.models import Document


def _sanitize_filename(source_id: str) -> str:
    """Replace filesystem-unsafe characters with underscores."""
    return re.sub(r'[/:*?"<>|\\]', "_", source_id)


def dump_documents(documents: list[Document], cache_dir: str) -> int:
    """Dump documents to disk as markdown files with YAML front-matter.

    Organizes by source_type: {cache_dir}/{source_type}/{sanitized_source_id}.md
    Clears each source_type subdirectory before writing so deleted docs don't linger.

    Returns the number of files written.
    """
    # Group documents by source_type
    by_source: dict[str, list[Document]] = {}
    for doc in documents:
        by_source.setdefault(doc.source_type, []).append(doc)

    count = 0
    for source_type, docs in by_source.items():
        type_dir = os.path.join(cache_dir, source_type)

        # Clear existing files in this source_type dir
        if os.path.isdir(type_dir):
            for f in os.listdir(type_dir):
                fp = os.path.join(type_dir, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        else:
            os.makedirs(type_dir, exist_ok=True)

        for doc in docs:
            filename = _sanitize_filename(doc.source_id) + ".md"
            filepath = os.path.join(type_dir, filename)

            front_matter_lines = [
                "---",
                f"source_id: {doc.source_id}",
                f"title: {doc.title}",
            ]
            if doc.url:
                front_matter_lines.append(f"url: {doc.url}")
            front_matter_lines.append(f"fetched_at: {doc.fetched_at.isoformat()}")
            front_matter_lines.append("---")

            with open(filepath, "w") as f:
                f.write("\n".join(front_matter_lines) + "\n" + doc.content)

            count += 1

    return count
