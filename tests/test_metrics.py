import pytest

from llm_eval_lab.metrics import answer_metrics, evidence_metrics, retrieval_metrics


def test_retrieval_metrics_found_at_first_position():
    result = retrieval_metrics(["a", "b"], ["a"], k=5)
    assert result["hit@5"] == 1.0
    assert result["recall@5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["precision@5"] == 0.2


def test_retrieval_metrics_uses_first_relevant_rank_and_unique_ids():
    result = retrieval_metrics(["x", "x", "b"], ["b", "c"], k=3)
    assert result["mrr"] == pytest.approx(1 / 2)
    assert result["recall@3"] == pytest.approx(0.5)


def test_retrieval_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must"):
        retrieval_metrics([], [], k=0)


def test_answer_metrics_normalizes_punctuation_and_case():
    result = answer_metrics("Hello, WORLD!", "hello world")
    assert result["exact_match"] == 1.0
    assert result["token_f1"] == 1.0


def test_empty_answer_has_zero_overlap():
    result = answer_metrics("", "a reference")
    assert result["token_f1"] == 0.0


def test_evidence_metrics_reports_limitations_and_citations():
    result = evidence_metrics("The policy is current", "The policy is current", ["doc-1"])
    assert result["has_citation"] is True
    assert result["citation_count"] == 1
    assert "do not establish" in result["warning"]
