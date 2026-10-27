"""Report serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json_report(report: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(report: dict[str, Any], destination: Path, title: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"- Cases: {report.get('case_count', 0)}"]
    if "passed_count" in report:
        lines.extend([f"- Passed: {report['passed_count']}", f"- Failed: {report['failed_count']}"])
    aggregate = report.get("aggregate")
    if aggregate is not None:
        lines.extend(["", "## Aggregate metrics", "", "| Metric | Mean |", "|---|---:|"])
        lines.extend(f"| {name} | {value:.4f} |" for name, value in sorted(aggregate.items()))
    if report.get("cases"):
        lines.extend(["", "## Cases", ""])
        for case in report["cases"]:
            status = "PASS" if case.get("passed", True) else "FAIL"
            lines.append(f"- **{case['id']}** — {status}")
            if case.get("failures"):
                lines.append(f"  - Failures: {', '.join(case['failures'])}")
            if case.get("metrics"):
                metrics = ", ".join(f"{key}={value:.3f}" for key, value in case["metrics"].items()
                                    if isinstance(value, (float, int)) and not isinstance(value, bool))
                if metrics:
                    lines.append(f"  - Metrics: {metrics}")
    if report.get("metric_notes"):
        lines.extend(["", "## Interpretation notes", ""])
        lines.extend(f"- {note}" for note in report["metric_notes"].values())
    if report.get("limitation"):
        lines.extend(["", "## Limitations", "", report["limitation"]])
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
