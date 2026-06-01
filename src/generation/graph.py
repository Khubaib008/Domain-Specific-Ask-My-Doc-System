"""LangGraph RAG workflow: Hybrid Retrieve → Rerank → Generate.

Phase 2 architecture:
    Question
       │
       ▼
  ┌─────────────┐
  │  Retrieve    │  Hybrid: ChromaDB vector + BM25 keyword (RRF fusion)
  │  (k=15)      │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Rerank      │  Cohere cross-encoder (top-5 from top-15)
  │  (top_n=5)   │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  Generate    │  LLM with strict citation enforcement
  └──────┬──────┘
         │
         ▼
      Answer
"""

import os
from typing import Annotated, TypedDict, Optional
from operator import add

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from src.generation.prompts import PromptManager
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CohereReranker

load_dotenv()


class RagState(TypedDict):
    """State passed between graph nodes."""
    question: str
    documents: Annotated[list[Document], add]
    reranked_documents: Annotated[list[Document], add]
    answer: str
    citations: list[dict]
    declined: bool
    retrieval_details: list[dict]


class HybridRetrieveNode:
    """Fetches documents via hybrid search (vector + BM25 with RRF fusion).

    Retrieves a larger initial candidate set (k=15 by default) to provide
    the reranker with a rich pool of candidates.
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        k: int = 15,
    ):
        self.retriever = hybrid_retriever or HybridRetriever()
        self.k = k

    def __call__(self, state: RagState) -> dict:
        docs = self.retriever.retrieve(state["question"], k=self.k)
        return {"documents": docs}


class RerankNode:
    """Rescores retrieved documents using Cohere's cross-encoder.

    Takes the top-k candidates from the hybrid retriever and returns
    only the top_n most relevant documents for generation.
    """

    def __init__(
        self,
        reranker: Optional[CohereReranker] = None,
        top_n: int = 5,
    ):
        self.reranker = reranker or CohereReranker(top_n=top_n)
        self.top_n = top_n

    def __call__(self, state: RagState) -> dict:
        docs = state.get("documents", [])
        if not docs:
            return {
                "reranked_documents": [],
                "retrieval_details": [],
            }

        reranked = self.reranker.rerank(
            query=state["question"],
            documents=docs,
            top_n=self.top_n,
        )

        reranked_docs = [doc for doc, _ in reranked]
        details = [
            {
                "source": doc.metadata.get("source", "?"),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "rerank_score": round(score, 4),
                "content_preview": doc.page_content[:150],
            }
            for doc, score in reranked
        ]

        return {
            "reranked_documents": reranked_docs,
            "retrieval_details": details,
        }


def _create_chat_llm() -> BaseLanguageModel:
    """Create the appropriate chat LLM based on environment config.

    Priority:
    1. Gemini (when GEMINI_API_KEY or GOOGLE_API_KEY is set)
    2. Azure OpenAI (when AZURE_OPENAI_ENDPOINT is set)
    3. Standard OpenAI / OpenRouter (fallback)
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
        print(f"[chat] Using Gemini: {model}")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=gemini_key,
            temperature=0,
        )

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=os.getenv("OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini"),
            temperature=0,
        )

    base_url = os.getenv("OPENAI_API_BASE") or None
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        openai_api_base=base_url,
    )


class GenerateNode:
    """Formats context, enforces citations, and generates the answer.

    Uses reranked_documents (from RerankNode) if available,
    otherwise falls back to documents (from RetrieveNode).
    """

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        if llm is None:
            self.llm = _create_chat_llm()
        else:
            self.llm = llm
        self.pm = prompt_manager or PromptManager()

    def _format_context(self, docs: list[Document]) -> tuple[str, list[dict]]:
        """Numbered context blocks for citation tracking."""
        parts = []
        citations = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", f"doc_{i}")
            chunk_idx = doc.metadata.get("chunk_index", i - 1)
            source_id = f"{source}#chunk{chunk_idx}"

            parts.append(
                f"[Document {i} | {source_id}]\n{doc.page_content}"
            )
            citations.append({
                "index": i,
                "source_id": source_id,
                "source": source,
                "chunk_index": chunk_idx,
                "page_content_preview": doc.page_content[:200],
                "metadata": dict(doc.metadata),
            })
        return "\n\n".join(parts), citations

    def __call__(self, state: RagState) -> dict:
        docs = state.get("reranked_documents") or state.get("documents", [])

        if not docs:
            return {
                "answer": self.pm.citation_declaration,
                "citations": [],
                "declined": True,
            }

        context, citations = self._format_context(docs)
        user_prompt = self.pm.generation_prompt(
            question=state["question"],
            context=context,
        )

        messages = [
            ("system", self.pm.system_prompt()),
            ("user", user_prompt),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        response = self.llm.invoke(prompt.format_messages())

        answer = response.content if hasattr(response, "content") else str(response)

        declined = self.pm.citation_declaration.strip().lower() in answer.strip().lower()

        return {
            "answer": answer,
            "citations": citations,
            "declined": declined,
        }


def build_rag_graph(
    vector_store: Optional[ChromaVectorStore] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
    reranker: Optional[CohereReranker] = None,
    llm: Optional[BaseLanguageModel] = None,
    retrieve_k: int = 15,
    rerank_top_n: int = 5,
    prompt_version: str = "v1",
    use_reranker: bool = True,
) -> StateGraph:
    """Build and compile the RAG LangGraph workflow.

    Phase 1 mode (use_reranker=False):
        Retrieve → Generate

    Phase 2 mode (use_reranker=True):
        HybridRetrieve → Rerank → Generate

    Args:
        vector_store: ChromaDB store.
        hybrid_retriever: Hybrid retriever (vector + BM25).
        reranker: Cohere cross-encoder reranker.
        llm: Language model.
        retrieve_k: Number of candidates from hybrid retriever.
        rerank_top_n: Number of final docs after reranking.
        prompt_version: Prompt YAML version.
        use_reranker: If False, falls back to Phase 1 pipeline.

    Returns:
        Compiled LangGraph StateGraph.
    """
    _pm = PromptManager(version=prompt_version)

    if use_reranker:
        _retriever = hybrid_retriever or HybridRetriever(vector_store=vector_store)
        _reranker = reranker or CohereReranker(top_n=rerank_top_n)

        retrieve_node = HybridRetrieveNode(hybrid_retriever=_retriever, k=retrieve_k)
        rerank_node = RerankNode(reranker=_reranker, top_n=rerank_top_n)
        generate_node = GenerateNode(llm=llm, prompt_manager=_pm)

        graph = StateGraph(RagState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("rerank", rerank_node)
        graph.add_node("generate", generate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "rerank")
        graph.add_edge("rerank", "generate")
        graph.add_edge("generate", END)
    else:
        _retriever = hybrid_retriever or HybridRetriever(vector_store=vector_store or ChromaVectorStore())
        retrieve_node = HybridRetrieveNode(
            hybrid_retriever=_retriever,
            k=rerank_top_n,
        )
        generate_node = GenerateNode(llm=llm, prompt_manager=_pm)

        graph = StateGraph(RagState)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("generate", generate_node)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

    return graph.compile()


def run_rag(
    question: str,
    vector_store: Optional[ChromaVectorStore] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
    reranker: Optional[CohereReranker] = None,
    retrieve_k: int = 15,
    rerank_top_n: int = 5,
    use_reranker: bool = True,
    **kwargs,
) -> dict:
    """Convenience function: build graph, invoke, return structured result.

    Returns:
        Dict with keys: answer, citations, declined, documents,
        reranked_documents, retrieval_details.
    """
    graph = build_rag_graph(
        vector_store=vector_store,
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        retrieve_k=retrieve_k,
        rerank_top_n=rerank_top_n,
        use_reranker=use_reranker,
        **kwargs,
    )
    result = graph.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "declined": result.get("declined", False),
        "num_docs_retrieved": len(result.get("documents", [])),
        "num_docs_reranked": len(result.get("reranked_documents", [])),
        "retrieval_details": result.get("retrieval_details", []),
    }
