"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from llm_eval_lab.compare import compare_reports
from llm_eval_lab.evaluator import evaluate_cases, validate_cases
from llm_eval_lab.reporting import write_json_report, write_markdown_report
from llm_eval_lab.safety import evaluate_safety_cases, validate_safety_cases


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _write_outputs(report: dict[str, Any], out_dir: Path, title: str) -> None:
    write_json_report(report, out_dir / "report.json")
    write_markdown_report(report, out_dir / "report.md", title)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval", description="Offline-first LLM and RAG evaluation tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate a RAG JSON dataset.")
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--out-dir", type=Path, default=Path("reports"))
    evaluate.add_argument("-k", type=int, default=5)
    safety = commands.add_parser("safety", help="Check recorded safety regression cases.")
    safety.add_argument("dataset", type=Path)
    safety.add_argument("--out-dir", type=Path, default=Path("reports"))
    compare = commands.add_parser("compare", help="Compare current report to a baseline.")
    compare.add_argument("current", type=Path)
    compare.add_argument("baseline", type=Path)
    compare.add_argument("--max-regression", type=float, default=0.05)
    compare.add_argument("--out", type=Path, default=Path("reports/comparison.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "evaluate":
            if args.k < 1:
                raise ValueError("--k must be >= 1")
            report = evaluate_cases(validate_cases(_read_json(args.dataset)), k=args.k)
            _write_outputs(report, args.out_dir, "RAG Evaluation Report")
            print(f"Evaluated {report['case_count']} cases; reports written to {args.out_dir}")
            return 0
        if args.command == "safety":
            report = evaluate_safety_cases(validate_safety_cases(_read_json(args.dataset)))
            _write_outputs(report, args.out_dir, "Safety Regression Report")
            print(f"Safety checks: {report['passed_count']} passed, {report['failed_count']} failed.")
            return 0 if report["failed_count"] == 0 else 1
        current, baseline = _read_json(args.current), _read_json(args.baseline)
        report = compare_reports(current, baseline, args.max_regression)
        write_json_report(report, args.out)
        print(f"Baseline comparison {'passed' if report['passed'] else 'failed'}.")
        if not report["passed"]:
            print(f"Regressions: {report['regressions']}; missing metrics: {report['missing_metrics']}")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
