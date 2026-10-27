import pytest

from llm_eval_lab.evaluator import evaluate_cases, validate_cases


def test_evaluation_aggregates_only_available_metrics():
    cases = validate_cases([{
        "id": "case-1",
        "answer": "same answer",
        "reference_answer": "same answer",
        "retrieved_doc_ids": ["doc"],
        "relevant_doc_ids": ["doc"]
    }])
    report = evaluate_cases(cases)
    assert report["case_count"] == 1
    assert report["aggregate"]["exact_match"] == 1.0
    assert report["aggregate"]["mrr"] == 1.0


def test_dataset_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cases([{"id": "same"}, {"id": "same"}])


def test_dataset_rejects_wrong_field_type():
    with pytest.raises(ValueError, match="answer must be a string"):
        validate_cases([{"id": "x", "answer": 7}])


def test_dataset_requires_array():
    with pytest.raises(ValueError, match="JSON array"):
        validate_cases({"id": "x"})
