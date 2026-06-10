from __future__ import annotations

import json
from pathlib import Path

from agent_evals.cli import main
from agent_evals.comparisons import CompareOptions, compare_runs
from agent_evals.runners import RunOptions, run_eval


def test_compare_detects_regression_and_gate_failure(tmp_path):
    baseline_cases = _write_cases(tmp_path / "baseline.eval_cases.jsonl", script="correct")
    candidate_cases = _write_cases(tmp_path / "candidate.eval_cases.jsonl", script="wrong_tool_args")

    baseline = run_eval(
        RunOptions(
            cases_path=baseline_cases,
            out_dir=tmp_path,
            run_id="baseline",
            rubric_path=None,
        )
    )
    candidate = run_eval(
        RunOptions(
            cases_path=candidate_cases,
            out_dir=tmp_path,
            run_id="candidate",
            rubric_path=None,
        )
    )

    compare_out = tmp_path / "compare"
    summary = compare_runs(
        CompareOptions(
            baseline_path=baseline.out_dir / "eval_results.jsonl",
            candidate_path=candidate.out_dir / "eval_results.jsonl",
            out_dir=compare_out,
            gates_path=Path("configs/gates.yaml"),
        )
    )

    assert summary.gate_triggered is True
    assert [case.case_id for case in summary.new_failures] == ["case_regresses"]
    assert summary.new_failures[0].candidate_failure_type == "tool_arguments"
    assert summary.metrics["tool_call_accuracy"].delta < 0
    assert [regression.tag for regression in summary.tag_regressions] == ["tool_call"]
    assert {failure.metric for failure in summary.gate_failures} == {"tool_call_accuracy"}

    json_payload = json.loads((compare_out / "compare_results.json").read_text())
    assert json_payload["gate_triggered"] is True
    assert json_payload["new_failures"][0]["case_id"] == "case_regresses"

    markdown = (compare_out / "compare_report.md").read_text(encoding="utf-8")
    assert "case_regresses" in markdown
    assert "Gate triggered." in markdown
    assert "tool_call_accuracy" in markdown

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline.out_dir / "eval_results.jsonl"),
            "--candidate",
            str(candidate.out_dir / "eval_results.jsonl"),
            "--out",
            str(tmp_path / "cli_compare"),
        ]
    )
    assert exit_code == 1


def test_compare_returns_zero_when_candidate_does_not_regress(tmp_path):
    baseline_cases = _write_cases(tmp_path / "baseline.eval_cases.jsonl", script="correct")
    candidate_cases = _write_cases(tmp_path / "candidate.eval_cases.jsonl", script="correct")

    baseline = run_eval(
        RunOptions(
            cases_path=baseline_cases,
            out_dir=tmp_path,
            run_id="baseline_ok",
            rubric_path=None,
        )
    )
    candidate = run_eval(
        RunOptions(
            cases_path=candidate_cases,
            out_dir=tmp_path,
            run_id="candidate_ok",
            rubric_path=None,
        )
    )

    exit_code = main(
        [
            "compare",
            "--baseline",
            str(baseline.out_dir / "eval_results.jsonl"),
            "--candidate",
            str(candidate.out_dir / "eval_results.jsonl"),
            "--out",
            str(tmp_path / "cli_compare_ok"),
            "--fail-if-drop",
            "tool_call_accuracy=0.05",
        ]
    )

    assert exit_code == 0


def test_compare_reports_candidate_only_failed_cases(tmp_path):
    baseline_path = _write_results(
        tmp_path / "baseline.eval_results.jsonl",
        [_result("old_pass", passed=True, aggregate_score=1.0)],
    )
    candidate_path = _write_results(
        tmp_path / "candidate.eval_results.jsonl",
        [
            _result("old_pass", passed=True, aggregate_score=1.0),
            _result("new_fail", passed=False, aggregate_score=0.0),
        ],
    )

    summary = compare_runs(
        CompareOptions(
            baseline_path=baseline_path,
            candidate_path=candidate_path,
            out_dir=tmp_path / "compare_candidate_only",
            gates_path=None,
        )
    )

    assert [case.case_id for case in summary.new_failures] == ["new_fail"]
    assert summary.new_failures[0].baseline_failure_type == "missing"
    markdown = (tmp_path / "compare_candidate_only" / "compare_report.md").read_text(
        encoding="utf-8"
    )
    assert "new_fail" in markdown


def _write_cases(path: Path, *, script: str) -> Path:
    rows = [
        {
            "id": "case_regresses",
            "input": {"messages": [{"role": "user", "content": "Check order A123."}]},
            "expected": {
                "answer_contains": ["order"],
                "tool_calls": [
                    {
                        "tool_name": "lookup_order",
                        "arguments": {"order_id": "A123"},
                        "match_mode": "strict",
                    }
                ],
                "outcome": {"task_success": True},
            },
            "metadata": {"tags": ["tool_call"], "mock": {"script": script}},
        },
        {
            "id": "case_stable",
            "input": {"messages": [{"role": "user", "content": "Say hello."}]},
            "expected": {
                "answer_contains": ["hello"],
                "outcome": {"task_success": True},
            },
            "metadata": {
                "tags": ["answer_rule"],
                "mock": {"script": "correct", "final_answer": "hello from mock agent"},
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _write_results(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _result(case_id: str, *, passed: bool, aggregate_score: float) -> dict:
    failure_type = "" if passed else "task_failure"
    reason = "" if passed else "candidate-only case failed"
    return {
        "case_id": case_id,
        "pass": passed,
        "aggregate_score": aggregate_score,
        "scores": {
            "task_success": aggregate_score,
            "tool_call_accuracy": 1.0,
            "final_answer_correctness": aggregate_score,
            "safety": 1.0,
        },
        "failure_type": failure_type,
        "reason": reason,
        "trace_path": f"{case_id}.trace.json",
        "latency_ms": 0,
        "cost_usd": 0.0,
        "tags": [],
        "score_results": [],
    }
