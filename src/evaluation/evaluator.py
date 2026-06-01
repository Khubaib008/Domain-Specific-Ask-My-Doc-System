"""Ragas-based RAG evaluation script.

Measures:
  - faithfulness: Are answers grounded in retrieved chunks?
  - answer_relevancy: How relevant is the answer to the question?

Usage:
    from src.evaluation.evaluator import RagEvaluator
    evaluator = RagEvaluator(rag_pipeline_func)
    results = evaluator.evaluate(golden_dataset_df)
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class RagEvaluator:
    """Evaluates a RAG pipeline using Ragas metrics.

    Args:
        rag_pipeline_func: A callable that takes a question string and returns
            a dict with keys: "answer", "documents" (list of langchain Documents).
    """

    def __init__(
        self,
        rag_pipeline_func: Callable[[str], dict],
    ):
        self.rag_func = rag_pipeline_func

    def _run_pipeline_on_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the RAG pipeline on every question in the dataset.

        Returns the input DataFrame augmented with "answer" and "contexts" columns.
        """
        answers = []
        contexts = []

        for _, row in df.iterrows():
            question = row["question"]
            try:
                result = self.rag_func(question)
                answers.append(result.get("answer", ""))

                docs = result.get("documents", [])
                ctx = [doc.page_content for doc in docs]
                contexts.append(ctx)
            except Exception as e:
                print(f"  [ERROR] Question '{question[:50]}...': {e}")
                answers.append("")
                contexts.append([])

        df = df.copy()
        df["answer"] = answers
        df["contexts"] = contexts
        return df

    def evaluate(
        self,
        golden_df: pd.DataFrame,
        metrics: Optional[list[str]] = None,
    ) -> dict:
        """Run Ragas evaluation on the golden dataset.

        Args:
            golden_df: DataFrame with columns: question, reference_answer.
            metrics: List of metric names. Defaults to ["faithfulness", "answer_relevancy"].

        Returns:
            Dict with metric names as keys and scores as values.
        """
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }

        selected_metrics = []
        for m in (metrics or ["faithfulness", "answer_relevancy"]):
            if m in metric_map:
                selected_metrics.append(metric_map[m])
            else:
                print(f"  [WARN] Unknown metric '{m}', skipping")

        print(f"Running RAG pipeline on {len(golden_df)} questions...")
        results_df = self._run_pipeline_on_dataset(golden_df)

        ragas_data = {
            "question": results_df["question"].tolist(),
            "answer": results_df["answer"].tolist(),
            "contexts": results_df["contexts"].tolist(),
            "ground_truth": results_df["reference_answer"].tolist(),
        }

        dataset = Dataset.from_dict(ragas_data)

        print(f"Computing Ragas metrics: {[m.name for m in selected_metrics]}...")
        result = ragas_evaluate(dataset, metrics=selected_metrics)

        scores = {m.name: result[m.name] for m in selected_metrics}
        return scores

    def evaluate_and_report(
        self,
        golden_df: pd.DataFrame,
        faithfulness_threshold: float = 0.85,
    ) -> bool:
        """Evaluate and print a pass/fail report.

        Returns:
            True if all metrics pass their thresholds, False otherwise.
        """
        scores = self.evaluate(golden_df)

        print("\n" + "=" * 60)
        print("  RAG EVALUATION REPORT")
        print("=" * 60)

        all_passed = True
        for metric_name, score in scores.items():
            threshold = faithfulness_threshold if metric_name == "faithfulness" else 0.7
            passed = score >= threshold
            status = "PASS" if passed else "FAIL"
            print(f"  {metric_name:25s}: {score:.4f} (threshold: {threshold}) [{status}]")
            if not passed:
                all_passed = False

        print("=" * 60)
        if all_passed:
            print("  OVERALL: PASS")
        else:
            print("  OVERALL: FAIL — scores below threshold")
        print("=" * 60)

        return all_passed
