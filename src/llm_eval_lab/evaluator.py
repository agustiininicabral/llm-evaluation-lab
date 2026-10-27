"""Dataset validation and evaluation orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from llm_eval_lab.metrics import answer_metrics, evidence_metrics, retrieval_metrics


def validate_cases(cases: Any) -> list[dict[str, Any]]:
    """Validate a JSON-decoded list of evaluation cases."""
    if not isinstance(cases, list):
        raise ValueError("Dataset root must be a JSON array.")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Case at index {index} must be a JSON object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Case at index {index} must have a non-empty string id.")
        if case_id in seen:
            raise ValueError(f"Duplicate case id: {case_id}")
        seen.add(case_id)
        for field in ("question", "answer", "reference_answer", "evidence"):
            if field in case and not isinstance(case[field], str):
                raise ValueError(f"Case {case_id}: {field} must be a string.")
        for field in ("retrieved_doc_ids", "relevant_doc_ids", "citations"):
            if field in case and (not isinstance(case[field], list) or not all(isinstance(item, str) for item in case[field])):
                raise ValueError(f"Case {case_id}: {field} must be a list of strings.")
    return cases


def evaluate_cases(cases: Sequence[Mapping[str, Any]], k: int = 5) -> dict[str, Any]:
    """Evaluate cases and aggregate available numeric metrics by arithmetic mean."""
    if k < 1:
        raise ValueError("k must be >= 1")
    results: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {}
    for case in cases:
        metrics: dict[str, Any] = {}
        if "relevant_doc_ids" in case:
            metrics.update(retrieval_metrics(case.get("retrieved_doc_ids", []), case["relevant_doc_ids"], k=k))
        if "reference_answer" in case and "answer" in case:
            metrics.update(answer_metrics(case["answer"], case["reference_answer"]))
        if "evidence" in case and "answer" in case:
            metrics.update(evidence_metrics(case["answer"], case["evidence"], case.get("citations", [])))
        for name, value in metrics.items():
            if isinstance(value, (float, int)) and not isinstance(value, bool):
                metric_values.setdefault(name, []).append(float(value))
        results.append({"id": case["id"], "metrics": metrics})
    aggregate = {name: sum(values) / len(values) for name, values in sorted(metric_values.items()) if values}
    return {
        "schema_version": 1,
        "case_count": len(results),
        "aggregate": aggregate,
        "cases": results,
        "metric_notes": {
            "answer": "Token overlap is lexical, not semantic similarity.",
            "evidence": "Evidence overlap is a heuristic and cannot verify entailment.",
            "retrieval": "Metrics depend on the supplied relevant document IDs.",
        },
    }
