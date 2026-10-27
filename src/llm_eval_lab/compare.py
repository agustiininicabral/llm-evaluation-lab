"""Compare aggregate evaluation reports against a baseline."""

from __future__ import annotations

from typing import Any


def compare_reports(current: dict[str, Any], baseline: dict[str, Any], max_regression: float = 0.05) -> dict[str, Any]:
    """Flag higher-is-better metrics that fell by more than max_regression."""
    if max_regression < 0:
        raise ValueError("max_regression must be >= 0")
    current_metrics = current.get("aggregate", {})
    baseline_metrics = baseline.get("aggregate", {})
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
        "note": "Absolute deltas are meaningful only when datasets and metric definitions match.",
    }
