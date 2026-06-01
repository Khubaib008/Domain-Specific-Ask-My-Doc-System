"""Pydantic schemas for the RAG API request/response models."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer using the RAG pipeline.",
        examples=["What is machine learning?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Override: number of documents to retrieve before reranking.",
    )
    rerank_top_n: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Override: number of final documents after reranking.",
    )


class Citation(BaseModel):
    """A single citation referencing a source document."""
    index: int
    source_id: str
    source: str
    chunk_index: int
    page_content_preview: str


class RetrievalDetail(BaseModel):
    """Debug info for a reranked document."""
    source: str
    chunk_index: int
    rerank_score: float
    content_preview: str


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""
    question: str
    answer: str
    citations: list[Citation]
    declined: bool
    num_docs_retrieved: int = Field(..., alias="num_docs_retrieved")
    num_docs_reranked: int = Field(..., alias="num_docs_reranked")
    retrieval_details: list[RetrievalDetail] = []


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    version: str
    vector_store_count: int
    bm25_built: bool
