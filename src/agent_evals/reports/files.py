"""Markdown and CSV report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from agent_evals.traces.schema import EvalCase


def write_summary_csv(path: Path, records: list) -> None:
    fieldnames = [
        "case_id",
        "pass",
        "aggregate_score",
        "task_success",
        "tool_call_accuracy",
        "trajectory_score",
        "final_answer_correctness",
        "failure_type",
        "latency_ms",
        "cost_usd",
        "tags",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "case_id": record.case_id,
                    "pass": record.passed,
                    "aggregate_score": f"{record.aggregate_score:.4f}",
                    "task_success": _score(record, "task_success"),
                    "tool_call_accuracy": _score(record, "tool_call_accuracy"),
                    "trajectory_score": _score(record, "tool_call_accuracy"),
                    "final_answer_correctness": _score(record, "final_answer_correctness"),
                    "failure_type": record.failure_type,
                    "latency_ms": record.latency_ms,
                    "cost_usd": f"{record.cost_usd:.6f}",
                    "tags": ",".join(record.tags),
                }
            )


def write_eval_report(path: Path, run_id: str, records: list) -> None:
    summary = _summary(records)
    lines = [
        "# Eval Report",
        "",
        "## Run Metadata",
        "",
        f"- Run id: `{run_id}`",
        f"- Case count: {summary['case_count']}",
        "",
        "## Overall Metrics",
        "",
        f"- Pass rate: {summary['pass_rate']:.1%} ({summary['pass_count']}/{summary['case_count']})",
        f"- Average latency: {summary['avg_latency_ms']:.1f} ms",
        f"- Average cost: ${summary['avg_cost_usd']:.6f}",
        f"- Average tool call accuracy: {_avg_score(records, 'tool_call_accuracy'):.3f}",
        f"- Average final answer correctness: {_avg_score(records, 'final_answer_correctness'):.3f}",
        "",
        "## Case Results",
        "",
        "| Case | Result | Aggregate | Failure | Reason |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        result = "PASS" if record.passed else "FAIL"
        lines.append(
            f"| `{record.case_id}` | {result} | {record.aggregate_score:.3f} | "
            f"{record.failure_type} | {_escape_table(record.reason)} |"
        )

    lines.extend(["", "## Tag Slices", ""])
    if summary["tag_slices"]:
        lines.extend(["| Tag | Cases | Pass rate |", "| --- | ---: | ---: |"])
        for tag, tag_summary in summary["tag_slices"].items():
            lines.append(
                f"| `{tag}` | {tag_summary['case_count']} | {tag_summary['pass_rate']:.1%} |"
            )
    else:
        lines.append("No tags recorded.")

    lines.extend(["", "## Failure Type Distribution", ""])
    if summary["failure_counts"]:
        lines.extend(["| Failure type | Count |", "| --- | ---: |"])
        for failure_type, count in summary["failure_counts"].items():
            lines.append(f"| `{failure_type}` | {count} |")
    else:
        lines.append("No failures.")

    lines.extend(
        [
            "",
            "## Top Regressions",
            "",
            "Baseline comparison is not implemented in this phase.",
            "",
            "## Suggested Next Steps",
            "",
            "- Inspect failed cases in `failed_cases.md`.",
            "- Calibrate weights and thresholds in Phase 7 after collecting real runs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_failed_cases(
    path: Path,
    cases: list[EvalCase],
    records: list,
    traces_dir: Path,
) -> None:
    cases_by_id = {case.id: case for case in cases}
    failed = [record for record in records if not record.passed]
    if not failed:
        path.write_text("# Failed Cases\n\nNo failed cases.\n", encoding="utf-8")
        return

    lines = ["# Failed Cases", ""]
    for record in failed:
        case = cases_by_id[record.case_id]
        trace_path = traces_dir / f"{record.case_id}.json"
        trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                f"## Case: {record.case_id}",
                "",
                "Result: FAIL",
                f"Failure: {record.failure_type}",
                "",
                "Input:",
                _case_input(case),
                "",
                "Trace:",
                *_trace_summary(trace_payload),
                "",
                "Reason:",
                record.reason,
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _score(record, name: str) -> float:
    return record.scores.get(name, 0.0)


def _avg_score(records: list, name: str) -> float:
    if not records:
        return 0.0
    return sum(_score(record, name) for record in records) / len(records)


def _summary(records: list) -> dict:
    from agent_evals.runners.pipeline import summarize_records

    return summarize_records(records)


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _case_input(case: EvalCase) -> str:
    return "\n".join(str(message.content) for message in case.input.messages)


def _trace_summary(trace_payload: dict) -> list[str]:
    lines: list[str] = []
    for index, step in enumerate(trace_payload.get("steps", []), 1):
        step_type = step.get("type")
        if step_type == "tool_call" and step.get("tool_call"):
            tool_call = step["tool_call"]
            lines.append(
                f"{index}. [Tool] {tool_call['tool_name']}({json.dumps(tool_call.get('arguments', {}), ensure_ascii=False)})"
            )
        elif step_type == "observation":
            lines.append(f"{index}. [Observation] {step.get('summary', '')}")
        elif step_type == "llm":
            lines.append(f"{index}. [LLM] {step.get('summary', '')}")
        else:
            lines.append(f"{index}. [{step_type}] {step.get('summary', '')}")
    return lines
