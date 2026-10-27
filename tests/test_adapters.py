import pytest

from llm_eval_lab.adapters import (
    adapt_anthropic_message,
    adapt_openai_chat_completion,
    adapt_text_output,
)
from llm_eval_lab.evaluator import validate_cases


def test_adapts_generic_recorded_output_and_preserves_evaluation_metadata():
    case = adapt_text_output(
        "The answer is 42.",
        case_id="  case-1 ",
        question="What is the answer?",
        reference_answer="42",
        retrieved_doc_ids=("doc-1",),
        relevant_doc_ids=("doc-1", "doc-2"),
        evidence="The answer is 42.",
        citations=("doc-1",),
    )

    assert case == {
        "id": "case-1",
        "question": "What is the answer?",
        "answer": "The answer is 42.",
        "reference_answer": "42",
        "retrieved_doc_ids": ["doc-1"],
        "relevant_doc_ids": ["doc-1", "doc-2"],
        "evidence": "The answer is 42.",
        "citations": ["doc-1"],
    }
    assert validate_cases([case]) == [case]


def test_adapts_openai_compatible_chat_completion():
    case = adapt_openai_chat_completion(
        {
            "id": "chatcmpl-example",
            "choices": [
                {"message": {"role": "assistant", "content": "A recorded response."}}
            ],
        },
        case_id="openai-1",
        question="Question",
    )

    assert case["answer"] == "A recorded response."
    assert case["id"] == "openai-1"


def test_adapts_anthropic_text_blocks_and_ignores_non_text_blocks():
    case = adapt_anthropic_message(
        {
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "tool_use", "id": "tool-1", "name": "search", "input": {}},
                {"type": "text", "text": "Second part."},
            ]
        },
        case_id="anthropic-1",
        question="Question",
    )

    assert case["answer"] == "First part.\nSecond part."


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "choices list"),
        ({"choices": []}, "choices list"),
        ({"choices": ["not-an-object"]}, "choice must be an object"),
        ({"choices": [{"message": None}]}, "message object"),
        ({"choices": [{"message": {"content": None}}]}, "content must be a string"),
    ],
)
def test_openai_adapter_rejects_malformed_response_shapes(response, message):
    with pytest.raises(ValueError, match=message):
        adapt_openai_chat_completion(response, case_id="x", question="Question")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ({}, "content list"),
        ({"content": []}, "does not contain any text blocks"),
        ({"content": [{"type": "text", "text": 123}]}, "must contain string text"),
    ],
)
def test_anthropic_adapter_rejects_malformed_response_shapes(response, message):
    with pytest.raises(ValueError, match=message):
        adapt_anthropic_message(response, case_id="x", question="Question")


def test_adapter_rejects_invalid_metadata_and_field_types():
    with pytest.raises(ValueError, match="case_id"):
        adapt_text_output("answer", case_id=" ", question="Question")
    with pytest.raises(ValueError, match="question"):
        adapt_text_output("answer", case_id="case-1", question=" ")
    with pytest.raises(ValueError, match="sequence of strings"):
        adapt_text_output(
            "answer",
            case_id="case-1",
            question="Question",
            citations=["doc-1", 2],
        )
