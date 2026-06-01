"""FastAPI server wrapping the LangGraph RAG pipeline.

Endpoints:
    GET  /health          — Service health + index stats
    POST /ask             — Run the full RAG pipeline (hybrid retrieve → rerank → generate)
    POST /ask/baseline    — Run Phase 1 baseline (vector retrieve → generate, no rerank)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import AskRequest, AskResponse, HealthResponse
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CohereReranker
from src.generation.graph import build_rag_graph, RagState

load_dotenv()

vector_store: Optional[ChromaVectorStore] = None
hybrid_retriever: Optional[HybridRetriever] = None
reranker: Optional[CohereReranker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup, clean up on shutdown."""
    global vector_store, hybrid_retriever, reranker

    vector_store = ChromaVectorStore()
    hybrid_retriever = HybridRetriever(vector_store=vector_store)

    if vector_store.count > 0:
        all_docs = vector_store.search(" ", k=vector_store.count)
        hybrid_retriever.build_bm25(all_docs)

    if os.getenv("COHERE_API_KEY"):
        reranker = CohereReranker()

    yield


app = FastAPI(
    title="RAG System API",
    description="Production-grade RAG with hybrid retrieval, Cohere reranking, and strict citations.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with index statistics."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        vector_store_count=vector_store.count if vector_store else 0,
        bm25_built=hybrid_retriever.bm25.is_built if hybrid_retriever else False,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Run the full Phase 2 RAG pipeline: hybrid retrieve → rerank → generate.

    Returns the answer with strict citations from the reranked documents.
    If the model cannot answer from the context, `declined` will be true.
    """
    if not vector_store or vector_store.count == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Ingest documents before querying.",
        )

    use_reranker = reranker is not None
    retrieve_k = request.top_k or 15
    rerank_top_n = request.rerank_top_n or 5

    try:
        graph = build_rag_graph(
            vector_store=vector_store,
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            retrieve_k=retrieve_k,
            rerank_top_n=rerank_top_n,
            use_reranker=use_reranker,
            prompt_version="v2",
        )
        result = graph.invoke({"question": request.question})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}",
        )

    return AskResponse(
        question=request.question,
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        declined=result.get("declined", False),
        num_docs_retrieved=len(result.get("documents", [])),
        num_docs_reranked=len(result.get("reranked_documents", [])),
        retrieval_details=result.get("retrieval_details", []),
    )


@app.post("/ask/baseline", response_model=AskResponse)
async def ask_baseline(request: AskRequest):
    """Run the Phase 1 baseline pipeline: vector retrieve → generate (no rerank).

    Useful for A/B comparison against the full Phase 2 pipeline.
    """
    if not vector_store or vector_store.count == 0:
        raise HTTPException(
            status_code=503,
            detail="Vector store is empty. Ingest documents before querying.",
        )

    k = request.rerank_top_n or 5

    try:
        graph = build_rag_graph(
            vector_store=vector_store,
            retrieve_k=k,
            use_reranker=False,
            prompt_version="v1",
        )
        result = graph.invoke({"question": request.question})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}",
        )

    return AskResponse(
        question=request.question,
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        declined=result.get("declined", False),
        num_docs_retrieved=len(result.get("documents", [])),
        num_docs_reranked=0,
        retrieval_details=[],
    )
