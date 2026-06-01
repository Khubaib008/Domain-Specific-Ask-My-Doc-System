"""Hybrid retriever: ensemble of ChromaDB vector search + BM25 keyword search."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from dotenv import load_dotenv

from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_index import BM25Index

load_dotenv()


class HybridRetriever:
    """Ensemble retriever combining dense (vector) and sparse (BM25) signals.

    Uses Reciprocal Rank Fusion (RRF) to merge rankings from both retrievers.
    This approach is parameter-free and avoids score normalization issues
    between different retrieval systems.

    Architecture:
        Query ──→ [Vector Search] ──┐
              └──→ [BM25 Search]  ──┘
                        │
                Reciprocal Rank Fusion
                        │
                  Top-k documents
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.vs = vector_store or ChromaVectorStore()
        self.bm25 = bm25_index or BM25Index()
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k 

    def build_bm25(self, documents: list[Document]) -> None:
        """Build or rebuild the BM25 index from documents."""
        self.bm25.build(documents)

    def save_bm25(self, path: Optional[str] = None) -> None:
        """Persist BM25 index to disk."""
        default = os.path.join(
            os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
            "bm25_index.pkl",
        )
        self.bm25.save(path or default)

    def load_bm25(self, path: Optional[str] = None) -> None:
        """Load BM25 index from disk."""
        default = os.path.join(
            os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
            "bm25_index.pkl",
        )
        self.bm25.load(path or default)


    def retrieve(
        self,
        query: str,
        k: int = 15,
        filter_dict: Optional[dict] = None,
    ) -> list[Document]:
        """Retrieve documents using hybrid search with RRF fusion.

        Args:
            query: The search query.
            k: Number of top results to return after fusion.
            filter_dict: Optional metadata filter for vector search.

        Returns:
            Top-k Documents ranked by RRF score.
        """
        vector_docs = self.vs.search(query, k=k * 2, filter_dict=filter_dict)
        vector_ranks = {id(doc): rank for rank, doc in enumerate(vector_docs, start=1)}

        bm25_results: list[tuple[Document, float]] = []
        if self.bm25.is_built:
            bm25_results = self.bm25.search(query, k=k * 2)
        bm25_ranks = {id(doc): rank for rank, (doc, _) in enumerate(bm25_results, start=1)}

        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())

        doc_lookup: dict[int, Document] = {}
        for doc in vector_docs:
            doc_lookup[id(doc)] = doc
        for doc, _ in bm25_results:
            doc_lookup[id(doc)] = doc

        rrf_scores: dict[int, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += self.vector_weight * (1.0 / (self.rrf_k + vector_ranks[doc_id]))
            if doc_id in bm25_ranks:
                score += self.bm25_weight * (1.0 / (self.rrf_k + bm25_ranks[doc_id]))
            rrf_scores[doc_id] = score

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [doc_lookup[doc_id] for doc_id in sorted_ids]

    def retrieve_with_details(
        self,
        query: str,
        k: int = 15,
    ) -> list[dict]:
        """Retrieve with per-signal rank details for debugging.

        Returns:
            List of dicts with document, rrf_score, vector_rank, bm25_rank.
        """
        vector_docs = self.vs.search(query, k=k * 2)
        vector_ranks = {id(doc): rank for rank, doc in enumerate(vector_docs, start=1)}

        bm25_results = self.bm25.search(query, k=k * 2) if self.bm25.is_built else []
        bm25_ranks = {id(doc): rank for rank, (doc, _) in enumerate(bm25_results, start=1)}

        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        doc_lookup: dict[int, Document] = {}
        for doc in vector_docs:
            doc_lookup[id(doc)] = doc
        for doc, _ in bm25_results:
            doc_lookup[id(doc)] = doc

        rrf_scores: dict[int, float] = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in vector_ranks:
                score += self.vector_weight * (1.0 / (self.rrf_k + vector_ranks[doc_id]))
            if doc_id in bm25_ranks:
                score += self.bm25_weight * (1.0 / (self.rrf_k + bm25_ranks[doc_id]))
            rrf_scores[doc_id] = score

        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
        return [
            {
                "document": doc_lookup[doc_id],
                "rrf_score": rrf_scores[doc_id],
                "vector_rank": vector_ranks.get(doc_id),
                "bm25_rank": bm25_ranks.get(doc_id),
            }
            for doc_id in sorted_ids
        ]
