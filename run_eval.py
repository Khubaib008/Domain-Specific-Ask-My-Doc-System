"""
RAG Evaluation Runner
=====================
Loads the golden dataset and runs Ragas evaluation against the RAG pipeline.

Usage:
    .venv-langchain/Scripts/python.exe run_eval.py
    .venv-langchain/Scripts/python.exe run_eval.py --threshold 0.85
    .venv-langchain/Scripts/python.exe run_eval.py --questions 10  # first 10 only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation.evaluator import RagEvaluator
from src.evaluation.golden_dataset import load_golden_dataset
from src.ingestion.loaders import load_markdown
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.graph import run_rag


def build_pipeline():
    """Build and return a RAG pipeline function for evaluation."""
    docs = load_markdown("data/sample_docs/ai_basics.md")
    docs += load_markdown("data/sample_docs/transformers.md")

    persist_dir = "./data/eval_chroma_db"
    vs = ChromaVectorStore(persist_dir=persist_dir, collection_name="eval")
    vs.add_documents(docs)
    print(f"Vector store: {vs.count} documents")

    hybrid = HybridRetriever(vector_store=vs)
    hybrid.build_bm25(docs)

    def pipeline(question: str) -> dict:
        return run_rag(
            question=question,
            hybrid_retriever=hybrid,
            use_reranker=False,
            prompt_version="v1",
        )

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--threshold", type=float, default=0.85, help="Faithfulness threshold")
    parser.add_argument("--questions", type=int, default=None, help="Limit to N questions")
    args = parser.parse_args()

    golden_df = load_golden_dataset()
    if args.questions:
        golden_df = golden_df.head(args.questions)
        print(f"Limited to first {args.questions} questions")

    pipeline = build_pipeline()

    evaluator = RagEvaluator(pipeline)
    passed = evaluator.evaluate_and_report(golden_df, faithfulness_threshold=args.threshold)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
