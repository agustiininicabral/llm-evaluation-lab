# Recorded-output adapters

The adapters convert captured provider responses into the provider-neutral case format accepted by `validate_cases` and `evaluate_cases`. They are pure data transformations: they do not make HTTP requests, invoke a model, import provider SDKs, or read credentials.

## Supported inputs

- `adapt_text_output(...)`: plain text captured from any local or hosted model runner.
- `adapt_openai_chat_completion(response, ...)`: a captured OpenAI-compatible Chat Completions JSON object with `choices[0].message.content` as a string.
- `adapt_anthropic_message(response, ...)`: a captured Messages API JSON object with text blocks in its `content` list. Text blocks are joined with newline characters; non-text blocks are ignored.

All adapters require a stable `case_id` and a non-empty `question`. Optional reference answer, retrieval IDs, evidence, and citations are preserved in the normalized case.

## Example

```python
from llm_eval_lab.adapters import adapt_openai_chat_completion
from llm_eval_lab.evaluator import evaluate_cases, validate_cases

captured_response = {
    "choices": [
        {"message": {"role": "assistant", "content": "The policy requires approval."}}
    ]
}

case = adapt_openai_chat_completion(
    captured_response,
    case_id="policy-001",
    question="What does the policy require?",
    reference_answer="Approval is required.",
    evidence="The policy requires approval.",
    citations=["policy-section-2"],
)

report = evaluate_cases(validate_cases([case]))
print(report["aggregate"])
```

## Boundaries and limitations

- The caller is responsible for capturing the response and providing the matching prompt, reference answer, retrieval IDs, evidence, and citations.
- The adapters do not verify provider signatures, model identity, request parameters, or whether the response belongs to the supplied case ID.
- The OpenAI-compatible adapter supports string message content only. Multimodal or structured content must be converted explicitly by the caller.
- The Anthropic adapter joins text blocks only. Tool-use blocks are intentionally not interpreted as user-facing answers.
- Adapter tests use synthetic objects and never contact providers. The normalized result should be stored with the original request/response and run metadata when auditability matters.
