# IGKE Modular RAG Backend

This is a modular refactor of the previous `backend.py`.

Pipeline:
User Query -> Router -> Retriever -> Reranker -> Generator -> Sources

The existing ChromaDB collection is reused. No re-indexing is required for the first migration.

## Run

Place this `backend.py` and the `rag/` folder beside your existing `ChromaDB/` and `frontend/` folders.

Set:
- `GROQ_API_KEY`

Optional:
- `CHROMA_PATH`
- `CHROMA_COLLECTION`
- `EMBEDDING_MODEL`
- `GROQ_MODEL`
- `RAG_TOP_K`
- `RAG_RERANK_ENABLED=true`
- `RAG_RERANKER=cross_encoder`

Then:
`uvicorn backend:app --reload --port 8000`

## Important

The first version keeps dense retrieval as the active implementation because the existing ChromaDB is already built around embeddings. The interfaces are intentionally separated so BM25/hybrid retrieval, alternative embedding models, alternative vector stores, query rewriting, and alternative LLMs can be added without rewriting the whole pipeline.
