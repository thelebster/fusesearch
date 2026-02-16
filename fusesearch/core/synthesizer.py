from fusesearch.llm.base import LLM

SYSTEM_PROMPT = """\
You are a search assistant. Answer the user's question based ONLY on the \
provided search results. Cite sources using [1], [2], etc. corresponding to \
the result numbers. If the search results don't contain enough information \
to answer the question, say so. Be concise and direct."""


def build_context(results: list[dict]) -> str:
    """Format search results into numbered context for the LLM."""
    parts = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "Untitled")
        heading = ""
        if result.get("heading_path"):
            heading = f" > {' > '.join(result['heading_path'])}"
        content = result.get("content", "")
        parts.append(f"[{i}] {title}{heading}\n{content}")
    return "\n\n---\n\n".join(parts)


def synthesize(llm: LLM, query: str, results: list[dict]) -> dict:
    """Generate an answer from search results using an LLM."""
    if not results:
        return {"answer": "No search results found.", "sources": []}

    context = build_context(results)
    user_prompt = f"Search results:\n\n{context}\n\nQuestion: {query}"
    answer = llm.complete(SYSTEM_PROMPT, user_prompt)

    sources = []
    for i, result in enumerate(results, 1):
        sources.append({
            "index": i,
            "title": result.get("title", "Untitled"),
            "source_type": result.get("source_type", ""),
            "heading_path": result.get("heading_path", []),
        })

    return {"answer": answer, "sources": sources}
