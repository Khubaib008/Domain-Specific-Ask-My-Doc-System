"""BM25 sparse retrieval index for keyword-based search."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Index:
    """BM25 keyword search index over a corpus of Documents.

    Builds an in-memory BM25Okapi index from tokenized document text.
    The index is serializable to disk for reuse across sessions.
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._docs: list[Document] = []
        self._tokenized: list[list[str]] = []


    def build(self, documents: list[Document]) -> "BM25Index":
        """Build the BM25 index from a list of Documents.

        Args:
            documents: The corpus to index.

        Returns:
            self (for chaining).
        """
        self._docs = documents
        self._tokenized = [self._tokenize(doc.page_content) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized)
        return self

    def update(self, documents: list[Document]) -> "BM25Index":
        """Rebuild the index with a new document set (alias for build)."""
        return self.build(documents)

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Search the BM25 index.

        Args:
            query: Search query string.
            k: Number of top results.

        Returns:
            List of (Document, bm25_score) tuples sorted by score desc.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        top_indices = np.argsort(scores)[::-1][:k]
        results = [
            (self._idx_to_doc(i), float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]
        return results

    def get_scores(self, query: str) -> np.ndarray:
        """Return raw BM25 scores for all documents."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")
        tokens = self._tokenize(query)
        return self._bm25.get_scores(tokens)

    def save(self, path: str | Path) -> None:
        """Persist the BM25 index to disk via pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "docs": self._docs,
                    "tokenized": self._tokenized,
                },
                f,
            )

    def load(self, path: str | Path) -> "BM25Index":
        """Load a previously saved BM25 index from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._docs = data["docs"]
        self._tokenized = data["tokenized"]
        self._bm25 = BM25Okapi(self._tokenized)
        return self

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer for BM25."""
        return text.lower().split()

    def _idx_to_doc(self, idx: int) -> Document:
        return self._docs[idx]

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None
