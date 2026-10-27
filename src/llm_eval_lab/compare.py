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
        if not math.isfinite(value):
            raise ValueError(f"{label} metric '{name}' must be finite.")
        validated[name] = float(value)
    return validated


def compare_reports(current: dict[str, Any], baseline: dict[str, Any], max_regression: float = 0.05) -> dict[str, Any]:
    """Flag higher-is-better metrics that fell by more than max_regression."""
    if isinstance(max_regression, bool) or not isinstance(max_regression, (int, float)):
        raise ValueError("max_regression must be a finite number >= 0")
    if not math.isfinite(max_regression) or max_regression < 0:
        raise ValueError("max_regression must be a finite number >= 0")
    current_metrics = _aggregate_metrics(current, "Current")
    baseline_metrics = _aggregate_metrics(baseline, "Baseline")
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
