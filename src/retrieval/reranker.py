"""Cohere cross-encoder reranker for rescoring retrieved documents."""

import os
from typing import Optional

import cohere
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class CohereReranker:
    """Cross-encoder reranker using Cohere's ReRank API.

    Takes a query and a list of documents, returns the documents
    reordered by relevance score from Cohere's cross-encoder model.

    The cross-encoder jointly encodes the query and each document,
    producing a more accurate relevance score than the bi-encoder
    (embedding) approach used in the initial retrieval stage.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        top_n: int = 5,
    ):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.model = model
        self.top_n = top_n
        self._client: Optional[cohere.Client] = None

    @property
    def client(self) -> cohere.Client:
        """Lazy-init the Cohere client."""
        if self._client is None:
            if not self.api_key:
                raise EnvironmentError(
                    "COHERE_API_KEY not set. Add it to .env or environment."
                )
            self._client = cohere.Client(self.api_key)
        return self._client

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: Optional[int] = None,
    ) -> list[tuple[Document, float]]:
        """Rerank documents by relevance to the query.

        Args:
            query: The user's question.
            documents: Candidate documents from the retrieval stage.
            top_n: Number of top results to return (overrides default).

        Returns:
            List of (Document, relevance_score) tuples sorted by score desc.
            Score is in [0, 1] where 1 is most relevant.
        """
        if not documents:
            return []

        n = top_n or self.top_n
        texts = [doc.page_content for doc in documents]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=min(n, len(texts)),
        )

        results = []
        for r in response.results:
            idx = r.index
            results.append((documents[idx], r.relevance_score))

        return results

    def rerank_with_scores(
        self,
        query: str,
        documents: list[Document],
        top_n: Optional[int] = None,
    ) -> list[dict]:
        """Rerank and return full details including original indices.

        Returns:
            List of dicts with document, score, original_index.
        """
        if not documents:
            return []

        n = top_n or self.top_n
        texts = [doc.page_content for doc in documents]

        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=texts,
            top_n=min(n, len(texts)),
        )

        return [
            {
                "document": documents[r.index],
                "score": r.relevance_score,
                "original_index": r.index,
            }
            for r in response.results
        ]
