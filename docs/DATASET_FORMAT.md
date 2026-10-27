# Dataset format

The CLI reads UTF-8 JSON arrays. Each record must be an object with a unique, non-empty string `id`. Optional fields are validated when present.

## RAG case fields

| Field | Type | Purpose |
|---|---|---|
| `id` | string, required | Stable case identifier |
| `question` | string | Input question, for traceability |
| `answer` | string | Recorded system answer |
| `reference_answer` | string | Reference used for lexical comparison |
| `retrieved_doc_ids` | string array | Ordered retrieval results |
| `relevant_doc_ids` | string array | Expected relevant document identifiers |
| `evidence` | string | Text supplied as evidence for the answer |
| `citations` | string array | Document IDs cited by the recorded answer |

Retrieval metrics are computed only when `relevant_doc_ids` is present. Missing retrieved IDs are treated as an empty list. Answer metrics require both `answer` and `reference_answer`. Evidence overlap requires `answer` and `evidence`.

## Safety case fields

| Field | Type | Purpose |
|---|---|---|
| `id` | string, required | Stable case identifier |
| `prompt` | string | Prompt associated with the recorded response |
| `response` | string | Response produced by the system under test |
| `should_refuse` | boolean | Whether a refusal is expected |
| `refusal_markers` | string array | Phrases used by the simple refusal heuristic |

Safety evaluation does not call a model. Supply captured outputs from your own test harness. Phrase matching can miss paraphrases, fail on multilingual outputs, and produce false positives; it is not a security guarantee.

## Metric definitions

- **hit@k:** 1 when any relevant document appears in the first k unique retrieved IDs, otherwise 0.
- **recall@k:** relevant retrieved IDs divided by all expected relevant IDs.
- **precision@k:** relevant retrieved IDs in the first k unique results divided by k.
- **MRR:** reciprocal rank of the first relevant result, or 0 when none is found.
- **exact match:** normalized strings match after case folding and punctuation removal.
- **token precision/recall/F1:** overlap of unique normalized tokens. These are lexical metrics, not semantic similarity.
- **evidence token overlap:** proportion of answer content tokens also present in the supplied evidence. It does not test entailment or detect unsupported inferences.
- **has citation:** at least one citation string is present. Citation validity is not verified.

## Reproducibility

Use the same dataset version, retrieval depth, normalization behavior, and metric definitions when comparing runs. The comparison gate assumes every metric is higher-is-better and uses an absolute delta threshold; do not include lower-is-better metrics without transforming or separately handling them. Reports must contain finite numeric aggregate values. The baseline file in examples is illustrative; regenerate it from a real current report before relying on regression thresholds.
