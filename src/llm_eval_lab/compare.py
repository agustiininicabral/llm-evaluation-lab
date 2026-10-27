"""Compare aggregate evaluation reports against a baseline."""

from __future__ import annotations

import math
from typing import Any


def _aggregate_metrics(report: Any, label: str) -> dict[str, float]:
    """Validate a report's aggregate mapping and return finite numeric metrics."""
    if not isinstance(report, dict):
        raise ValueError(f"{label} report must be a JSON object.")
    metrics = report.get("aggregate")
    if not isinstance(metrics, dict):
        raise ValueError(f"{label} report must contain an 'aggregate' object.")
    validated: dict[str, float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} metric names must be non-empty strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} metric '{name}' must be numeric.")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{label} metric '{name}' must be finite.")
        validated[name] = float(value)
    return validated


def _validate_provenance(
    current: dict[str, Any], baseline: dict[str, Any]
) -> list[str]:
    """Reject known dataset/setting mismatches and warn for legacy reports."""
    current_metadata = current.get("run_metadata")
    baseline_metadata = baseline.get("run_metadata")
    if not isinstance(current_metadata, dict) or not isinstance(baseline_metadata, dict):
        return [
            "Dataset provenance could not be fully verified because one or both reports lack run_metadata."
        ]

    for field in ("dataset_sha256", "mode", "retrieval_k"):
        current_value = current_metadata.get(field)
        baseline_value = baseline_metadata.get(field)
        if (
            current_value is not None
            and baseline_value is not None
            and current_value != baseline_value
        ):
            raise ValueError(
                f"Current and baseline reports have different {field}; "
                "compare reports generated from the same dataset and settings."
            )
    return []


def compare_reports(current: dict[str, Any], baseline: dict[str, Any], max_regression: float = 0.05) -> dict[str, Any]:
    """Flag higher-is-better metrics that fell by more than max_regression."""
    if isinstance(max_regression, bool) or not isinstance(max_regression, (int, float)):
        raise ValueError("max_regression must be a finite number >= 0")
    try:
        finite_threshold = math.isfinite(max_regression)
    except OverflowError:
        finite_threshold = False
    if not finite_threshold or max_regression < 0:
        raise ValueError("max_regression must be a finite number >= 0")
    current_metrics = _aggregate_metrics(current, "Current")
    baseline_metrics = _aggregate_metrics(baseline, "Baseline")
    provenance_warnings = _validate_provenance(current, baseline)
    comparisons: list[dict[str, Any]] = []
    missing: list[str] = []
    regressions: list[str] = []
    for name, baseline_value in sorted(baseline_metrics.items()):
        if name not in current_metrics:
            missing.append(name)
            continue
        current_value = current_metrics[name]
        delta = current_value - baseline_value
        regressed = delta < -max_regression
        if regressed:
            regressions.append(name)
        comparisons.append({"metric": name, "baseline": baseline_value, "current": current_value,
                            "delta": delta, "regression": regressed})
    return {
        "schema_version": 1, "max_regression": max_regression,
        "passed": not regressions and not missing, "regressions": regressions,
        "missing_metrics": missing, "comparisons": comparisons,
        "provenance_warnings": provenance_warnings,
        "note": "Absolute deltas are meaningful only when datasets and metric definitions match.",
    }
