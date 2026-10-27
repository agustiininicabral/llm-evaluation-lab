"""Deterministic metrics with explicit, documented limitations."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any


def normalize_text(text: str) -> str:
    """Lowercase text and collapse punctuation/whitespace for simple comparisons."""
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def retrieval_metrics(retrieved_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int = 5) -> dict[str, float]:
    """Calculate hit@k, recall@k, reciprocal rank, and precision@k.

    Duplicate retrieved IDs count once. Precision@k uses k as denominator even if fewer results exist.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1")
    relevant = set(relevant_doc_ids)
    retrieved = list(dict.fromkeys(retrieved_doc_ids))[:k]
    hits = [doc_id for doc_id in retrieved if doc_id in relevant]
    first_rank = next((rank for rank, doc_id in enumerate(retrieved, start=1) if doc_id in relevant), None)
    return {
        f"hit@{k}": float(bool(hits)),
        f"recall@{k}": len(hits) / len(relevant) if relevant else 0.0,
        "mrr": 1.0 / first_rank if first_rank else 0.0,
        f"precision@{k}": len(hits) / k,
    }


def answer_metrics(prediction: str, reference: str) -> dict[str, float]:
    """Calculate normalized exact match and token-overlap precision/recall/F1.

    Token overlap is lexical and is not a semantic equivalence score.
    """
    predicted_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    predicted_set, reference_set = set(predicted_tokens), set(reference_tokens)
    overlap = len(predicted_set & reference_set)
    precision = overlap / len(predicted_set) if predicted_set else 0.0
    recall = overlap / len(reference_set) if reference_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": float(normalize_text(prediction) == normalize_text(reference)),
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
    }


def evidence_metrics(answer: str, evidence: str, citations: Sequence[str]) -> dict[str, Any]:
    """Return lexical support signals, not a factuality verdict."""
    answer_tokens = {token for token in normalize_text(answer).split() if len(token) > 2}
    evidence_tokens = set(normalize_text(evidence).split())
    overlap = answer_tokens & evidence_tokens
    return {
        "evidence_token_overlap": len(overlap) / len(answer_tokens) if answer_tokens else 0.0,
        "has_citation": bool(citations),
        "citation_count": len(citations),
        "warning": "Lexical overlap and citation presence do not establish factual support.",
    }
