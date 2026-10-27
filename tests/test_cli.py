import json

from llm_eval_lab.cli import main


def test_evaluate_cli_writes_json_and_markdown(tmp_path):
    dataset = tmp_path / "cases.json"
    dataset.write_text(json.dumps([{
        "id": "one",
        "answer": "blue sky",
        "reference_answer": "blue sky",
        "retrieved_doc_ids": ["doc"],
        "relevant_doc_ids": ["doc"]
    }]), encoding="utf-8")
    output = tmp_path / "reports"
    assert main(["evaluate", str(dataset), "--out-dir", str(output)]) == 0
    assert (output / "report.json").exists()
    assert "Aggregate metrics" in (output / "report.md").read_text(encoding="utf-8")


def test_invalid_json_returns_user_facing_error(tmp_path):
    dataset = tmp_path / "bad.json"
    dataset.write_text("{broken", encoding="utf-8")
    assert main(["evaluate", str(dataset), "--out-dir", str(tmp_path / "out")]) == 2
