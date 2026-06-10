from __future__ import annotations

import json

import pytest

from agent_evals.datasets import DuplicateEvalCaseIdError, load_eval_cases_jsonl


def _case(case_id: str) -> dict:
    return {
        "id": case_id,
        "input": {
            "messages": [
                {"role": "user", "content": "Check order A123."},
            ]
        },
        "expected": {
            "answer_contains": ["order"],
            "tool_calls": [
                {
                    "tool_name": "lookup_order",
                    "arguments": {"order_id": "A123"},
                    "match_mode": "exact",
                }
            ],
        },
        "metadata": {
            "category": "customer_support",
            "difficulty": "easy",
            "tags": ["tool_call"],
        },
    }


def test_load_eval_cases_jsonl(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case("case_001")) + "\n", encoding="utf-8")

    cases = load_eval_cases_jsonl(path)

    assert len(cases) == 1
    assert cases[0].id == "case_001"
    assert cases[0].input.messages[0].role == "user"
    assert cases[0].expected.tool_calls[0].tool_name == "lookup_order"


def test_load_eval_cases_jsonl_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "duplicate_cases.jsonl"
    rows = [json.dumps(_case("case_001")), json.dumps(_case("case_001"))]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(DuplicateEvalCaseIdError, match="duplicate EvalCase id"):
        load_eval_cases_jsonl(path)
