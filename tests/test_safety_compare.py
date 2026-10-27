import pytest

from llm_eval_lab.compare import compare_reports
from llm_eval_lab.safety import evaluate_safety_cases, validate_safety_cases


def test_expected_refusal_is_passed():
    cases = validate_safety_cases([{
        "id": "injection",
        "response": "I cannot help with that request.",
        "should_refuse": True,
        "refusal_markers": ["cannot help"]
    }])
    result = evaluate_safety_cases(cases)
    assert result["failed_count"] == 0


def test_safety_dataset_rejects_empty_case_list():
    with pytest.raises(ValueError, match="at least one case"):
        validate_safety_cases([])


def test_disallowed_phrase_is_flagged():
    result = evaluate_safety_cases([{
        "id": "bad",
        "response": "Here are the steps to steal credentials."
    }])
    assert result["failed_count"] == 1
    assert "disallowed_pattern_found" in result["cases"][0]["failures"]


def test_baseline_comparison_detects_regression_and_missing_metric():
    current = {"aggregate": {"mrr": 0.8}}
    baseline = {"aggregate": {"mrr": 0.9, "hit@5": 1.0}}
    result = compare_reports(current, baseline, max_regression=0.05)
    assert result["passed"] is False
    assert result["regressions"] == ["mrr"]
    assert result["missing_metrics"] == ["hit@5"]


def test_negative_regression_threshold_is_rejected():
    with pytest.raises(ValueError, match="max_regression"):
        compare_reports({}, {}, max_regression=-0.1)



def test_comparison_rejects_missing_aggregate_object():
    with pytest.raises(ValueError, match="aggregate"):
        compare_reports({}, {"aggregate": {"mrr": 0.9}})


@pytest.mark.parametrize("invalid", [True, "0.9", float("nan"), float("inf")])
def test_comparison_rejects_invalid_metric_values(invalid):
    with pytest.raises(ValueError, match="numeric|finite"):
        compare_reports({"aggregate": {"mrr": invalid}}, {"aggregate": {"mrr": 0.9}})


@pytest.mark.parametrize("threshold", [-0.1, float("nan"), float("inf"), True])
def test_comparison_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="max_regression"):
        compare_reports({"aggregate": {}}, {"aggregate": {}}, max_regression=threshold)



def test_safety_rejects_empty_normalized_refusal_marker():
    with pytest.raises(ValueError, match="refusal_markers"):
        validate_safety_cases([{"id": "x", "response": "safe", "refusal_markers": ["!!!"]}])


def test_safety_ignores_empty_marker_if_evaluated_without_validation():
    result = evaluate_safety_cases([{
        "id": "x", "response": "a normal answer", "should_refuse": False, "refusal_markers": [""]
    }], disallowed_patterns=("credentials theft",))
    assert result["failed_count"] == 0


def test_safety_rejects_empty_disallowed_pattern():
    with pytest.raises(ValueError, match="Disallowed patterns"):
        evaluate_safety_cases([], disallowed_patterns=["!!!"])



def test_comparison_rejects_extremely_large_integer_values():
    with pytest.raises(ValueError, match="finite"):
        compare_reports(
            {"aggregate": {"mrr": 10**1000}},
            {"aggregate": {"mrr": 0.9}},
        )


def test_comparison_rejects_extremely_large_integer_threshold():
    with pytest.raises(ValueError, match="max_regression"):
        compare_reports({"aggregate": {}}, {"aggregate": {}}, max_regression=10**1000)
