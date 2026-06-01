"""ChromaDB vector store wrapper for ingestion and retrieval."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


def _create_embeddings():
    """Create the Gemini embedding client based on environment config."""
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not gemini_key:
        raise ValueError(
            "Gemini API key not found. Please set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file."
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    print(f"[embeddings] Using Gemini: {model}")
    
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=gemini_key,
    )


class ChromaVectorStore:
    """Thin wrapper around ChromaDB for document storage and retrieval.

    Manages a persistent local ChromaDB collection with OpenAI embeddings.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "rag_documents",
        embedding_model: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", "./data/chroma_db"
        )
        self.collection_name = collection_name

        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self._embeddings = _create_embeddings()

        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            documents: List of LangChain Documents to embed and store.

        Returns:
            List of document IDs assigned by ChromaDB.
        """
        return self._vectorstore.add_documents(documents)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[dict] = None,
    ) -> list[Document]:
        """Similarity search against the vector store.

        Args:
            query: The search query.
            k: Number of top results to return.
            filter_dict: Optional metadata filter.

        Returns:
            List of Documents sorted by relevance (closest first).
        """
        if filter_dict:
            return self._vectorstore.similarity_search(query, k=k, filter=filter_dict)
        return self._vectorstore.similarity_search(query, k=k)

    def search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[Document, float]]:
        """Similarity search returning documents with relevance scores.

        Returns:
            List of (Document, score) tuples, score in [0, 1].
        """
        return self._vectorstore.similarity_search_with_relevance_scores(query, k=k)

    def as_retriever(self, k: int = 5):
        """Return a LangChain retriever interface.

        Args:
            k: Default number of results.
        """
        return self._vectorstore.as_retriever(
            search_kwargs={"k": k},
        )

    @property
    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._vectorstore._collection.count()

    def clear(self):
        """Drop the entire collection. Useful for testing."""
        self._client.delete_collection(self.collection_name)
        self._vectorstore = Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
        )
