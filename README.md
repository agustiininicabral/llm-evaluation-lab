# LLM Evaluation Lab

An offline-first evaluation toolkit for retrieval-augmented generation (RAG) and LLM application regression tests. The core uses Python's standard library; no model provider, API key, Docker, or Ollama is required.

## Features

- Retrieval hit@k, recall@k, precision@k and reciprocal rank.
- Normalized answer matching and token-overlap metrics.
- Transparent lexical evidence-overlap and citation-presence signals.
- Phrase-based safety regression checks for recorded responses.
- Baseline comparison gates and JSON/Markdown reports.
- Pytest, Ruff and GitHub Actions CI.

Important: lexical and rule-based checks are regression signals, not proof of factual correctness or security. Semantic judgment requires human review and/or a separately configured evaluator.

## Quick start

Requires Python 3.10+.

    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"
    llm-eval evaluate examples/rag_cases.json --out-dir reports
    python -m pytest
    ruff check .

Run safety fixtures with `llm-eval safety examples/safety_cases.json --out-dir reports`. Compare reports with `llm-eval compare reports/report.json examples/baseline.json --max-regression 0.05`.

See docs/DATASET_FORMAT.md for the JSON schema. The core makes no network calls and does not invoke a model.

## Layout

- `src/llm_eval_lab/`: metrics, evaluator, safety checks, reports and CLI.
- `examples/`: synthetic datasets and baseline report.
- `tests/`: deterministic tests.
- `.github/workflows/`: automated checks.

## Roadmap

- [x] Offline retrieval and answer metrics
- [x] JSON and Markdown reports
- [x] Safety regression fixtures
- [x] Baseline regression gate
- [x] CI and tests
- [ ] Optional adapters for local and hosted model outputs
- [ ] Optional calibrated semantic judge

MIT License. See LICENSE.
