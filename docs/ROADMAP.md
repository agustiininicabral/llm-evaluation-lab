# Roadmap

## Initial offline release

- [x] Deterministic retrieval metrics (hit@k, recall@k, precision@k, MRR)
- [x] Normalized exact match and token-overlap answer metrics
- [x] Evidence-overlap and citation-presence indicators with explicit limitations
- [x] Recorded-response safety regression fixtures
- [x] JSON and Markdown output
- [x] Baseline comparison with configurable absolute regression threshold
- [x] Input validation and CLI exit codes
- [x] Unit tests and GitHub Actions matrix

## Next milestones

1. **Adapters:** define a stable interface for importing outputs from local inference servers and hosted providers without making either mandatory.
2. **Dataset quality:** add schema versioning, dataset metadata, and optional stratified summaries.
3. **Semantic evaluation:** add a pluggable judge interface with calibration examples, reproducibility metadata, and cost/time limits.
4. **Security coverage:** add application-level integration hooks and adversarial suites; keep phrase-based checks explicitly labeled as heuristics.
5. **Experiment comparison:** produce side-by-side Markdown/HTML reports with sample counts and confidence intervals where appropriate.

Each milestone should preserve offline tests and avoid implying that a single score certifies an LLM application as safe or correct.
