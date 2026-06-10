from __future__ import annotations

import csv
import json
from pathlib import Path

from agent_evals.runners import RunOptions, run_eval


SAMPLE_CASES = Path(__file__).resolve().parents[1] / "cases" / "sample.eval_cases.jsonl"


def test_mock_run_pipeline_writes_complete_run_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    summary = run_eval(
        RunOptions(
            cases_path=SAMPLE_CASES,
            out_dir=tmp_path,
            run_id="test_run",
        )
    )

    run_dir = tmp_path / "test_run"
    assert summary.out_dir == run_dir
    assert len(summary.records) == 5
    assert sum(1 for record in summary.records if not record.passed) >= 2

    eval_results_path = run_dir / "eval_results.jsonl"
    summary_path = run_dir / "summary.csv"
    report_path = run_dir / "eval_report.md"
    failed_cases_path = run_dir / "failed_cases.md"
    traces_dir = run_dir / "traces"

    assert eval_results_path.exists()
    assert summary_path.exists()
    assert report_path.exists()
    assert failed_cases_path.exists()
    assert traces_dir.exists()

    result_rows = [
        json.loads(line)
        for line in eval_results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(result_rows) == 5
    assert len(list(traces_dir.glob("*.json"))) == 5
    wrong_args_result = next(
        row for row in result_rows if row["case_id"] == "sample_wrong_tool_args"
    )
    assert wrong_args_result["failure_type"] == "tool_arguments"
    assert "expected order_id=A123, got A123_wrong" in wrong_args_result["reason"]

    failed_cases = failed_cases_path.read_text(encoding="utf-8")
    assert "sample_wrong_tool_args" in failed_cases
    assert "sample_missing_keyword" in failed_cases
    assert "Failure: tool_arguments" in failed_cases
    assert "Expected vs Actual:" in failed_cases
    assert 'lookup_order({"order_id": "A123"})' in failed_cases
    assert 'lookup_order({"order_id": "A123_wrong"})' in failed_cases
    assert "Trace:" in failed_cases
    assert "Reason:" in failed_cases

    report = report_path.read_text(encoding="utf-8")
    assert "Pass rate:" in report
    assert "sample_wrong_tool_args" in report
    assert "sample_missing_keyword" in report
    assert "tool_arguments" in report
    assert "Failure Type Distribution" in report
    assert "Tag Slices" in report
    assert "Judge skipped" in report

    with summary_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert {row["case_id"] for row in rows} == {row["case_id"] for row in result_rows}
