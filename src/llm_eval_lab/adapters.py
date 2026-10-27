"""Normalize captured model outputs into the lab's provider-neutral case format.

These adapters only transform data already captured by another runner. They do
not send requests, load models, or require provider SDKs or credentials.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _build_case(
    *,
    case_id: str,
    question: str,
    answer: str,
    reference_answer: str | None,
    retrieved_doc_ids: Sequence[str] | None,
    relevant_doc_ids: Sequence[str] | None,
    evidence: str | None,
    citations: Sequence[str] | None,
) -> dict[str, Any]:
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("case_id must be a non-empty string.")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if not isinstance(answer, str):
        raise ValueError("Model output text must be a string.")
    for field, value in (
        ("reference_answer", reference_answer),
        ("evidence", evidence),
    ):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be a string when supplied.")

    case: dict[str, Any] = {
        "id": case_id.strip(),
        "question": question,
        "answer": answer,
    }
    for field, value in (
        ("retrieved_doc_ids", retrieved_doc_ids),
        ("relevant_doc_ids", relevant_doc_ids),
        ("citations", citations),
    ):
        if value is not None:
            if isinstance(value, str) or not isinstance(value, Sequence):
                raise ValueError(f"{field} must be a sequence of strings.")
            if not all(isinstance(item, str) for item in value):
                raise ValueError(f"{field} must be a sequence of strings.")
            case[field] = list(value)

    if reference_answer is not None:
        case["reference_answer"] = reference_answer
    if evidence is not None:
        case["evidence"] = evidence
    return case


def adapt_text_output(
    output: str,
    *,
    case_id: str,
    question: str,
    reference_answer: str | None = None,
    retrieved_doc_ids: Sequence[str] | None = None,
    relevant_doc_ids: Sequence[str] | None = None,
    evidence: str | None = None,
    citations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Wrap a recorded local or hosted model output in a standard evaluation case."""
    return _build_case(
        case_id=case_id,
        question=question,
        answer=output,
        reference_answer=reference_answer,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        evidence=evidence,
        citations=citations,
    )


def adapt_openai_chat_completion(
    response: Mapping[str, Any],
    *,
    case_id: str,
    question: str,
    reference_answer: str | None = None,
    retrieved_doc_ids: Sequence[str] | None = None,
    relevant_doc_ids: Sequence[str] | None = None,
    evidence: str | None = None,
    citations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Extract text from a captured OpenAI-compatible chat-completion response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Chat-completion response must contain a non-empty choices list.")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("The first chat-completion choice must be an object.")
    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("The first chat-completion choice must contain a message object.")
    output = message.get("content")
    if not isinstance(output, str):
        raise ValueError("Chat-completion message content must be a string.")
    return adapt_text_output(
        output,
        case_id=case_id,
        question=question,
        reference_answer=reference_answer,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        evidence=evidence,
        citations=citations,
    )


def adapt_anthropic_message(
    response: Mapping[str, Any],
    *,
    case_id: str,
    question: str,
    reference_answer: str | None = None,
    retrieved_doc_ids: Sequence[str] | None = None,
    relevant_doc_ids: Sequence[str] | None = None,
    evidence: str | None = None,
    citations: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Extract text blocks from a captured Anthropic Messages API response."""
    content = response.get("content")
    if not isinstance(content, list):
        raise ValueError("Messages response must contain a content list.")

    text_blocks: list[str] = []
    for index, block in enumerate(content):
        if not isinstance(block, Mapping):
            raise ValueError(f"Content block at index {index} must be an object.")
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Text content block at index {index} must contain string text.")
            text_blocks.append(text)

    if not text_blocks:
        raise ValueError("Messages response does not contain any text blocks.")
    return adapt_text_output(
        "\n".join(text_blocks),
        case_id=case_id,
        question=question,
        reference_answer=reference_answer,
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
        evidence=evidence,
        citations=citations,
    )
