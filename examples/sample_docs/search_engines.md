# Search Engines and Information Retrieval

Search engines are systems designed to find information within large collections of data. They work by indexing content and then matching user queries against that index to return relevant results.

## How Search Works

The typical search pipeline has three stages: crawling (discovering content), indexing (organizing content for fast lookup), and retrieval (finding and ranking results for a query). Modern search systems combine multiple retrieval methods to improve result quality.

## Vector Search

Vector search (also called semantic search) converts text into numerical vectors using embedding models. Similar texts produce vectors that are close together in high-dimensional space. At query time, the search system finds stored vectors closest to the query vector using distance metrics like cosine similarity.

## Keyword Search (BM25)

BM25 is a classic ranking function used in keyword-based search. It scores documents based on term frequency (how often query terms appear) and inverse document frequency (how rare the terms are across all documents). BM25 excels at exact term matching but misses semantic relationships.

## Hybrid Search

Hybrid search combines vector and keyword approaches to get the best of both worlds. Vector search captures meaning and handles synonyms, while keyword search ensures exact matches are not missed. Reciprocal Rank Fusion (RRF) is a common technique to merge results from both methods into a single ranked list.

## Reranking

After initial retrieval, a cross-encoder reranker can rescore the top candidates. Unlike embedding models that encode query and document separately, cross-encoders process the (query, document) pair together, producing more accurate relevance scores at the cost of being slower. Reranking is typically applied to the top 10-50 candidates.
