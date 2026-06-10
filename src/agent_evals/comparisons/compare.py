"""Baseline versus candidate run comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from agent_evals.runners.pipeline import CaseRunRecord


class MetricChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: float
    candidate: float
    delta: float


class CaseChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    baseline_failure_type: str
    candidate_failure_type: str
    baseline_score: float
    candidate_score: float
    candidate_reason: str = ""


class TagRegression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta: float
    case_count: int


class GateFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline: float | None = None
    candidate: float
    drop: float | None = None
    threshold: float
    reason: str


class CompareSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_path: str
    candidate_path: str
    metrics: dict[str, MetricChange]
    new_failures: list[CaseChange]
    fixed_cases: list[CaseChange]
    tag_regressions: list[TagRegression]
    gate_triggered: bool
    gate_failures: list[GateFailure]


class CompareOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_path: Path
    candidate_path: Path
    out_dir: Path | None = None
    gates_path: Path | None = Path("configs/gates.yaml")
    fail_if_drop: dict[str, float] = Field(default_factory=dict)


def compare_runs(options: CompareOptions) -> CompareSummary:
    baseline = _load_records(options.baseline_path)
    candidate = _load_records(options.candidate_path)
    summary = _compare(
        baseline,
        candidate,
        baseline_path=options.baseline_path,
        candidate_path=options.candidate_path,
        gates=_load_gates(options.gates_path),
        fail_if_drop=options.fail_if_drop,
    )
    out_dir = options.out_dir or options.candidate_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "compare_results.json", summary)
    _write_markdown(out_dir / "compare_report.md", summary)
    return summary


def _load_records(path: Path) -> list[CaseRunRecord]:
    records: list[CaseRunRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(CaseRunRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid eval result at {path}:{line_number}: {exc}") from exc
    return records


def _compare(
    baseline: list[CaseRunRecord],
    candidate: list[CaseRunRecord],
    *,
    baseline_path: Path,
    candidate_path: Path,
    gates: dict[str, dict[str, float]],
    fail_if_drop: dict[str, float],
) -> CompareSummary:
    baseline_by_id = {record.case_id: record for record in baseline}
    candidate_by_id = {record.case_id: record for record in candidate}
    common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))

    metrics = _metric_changes(baseline, candidate)
    new_failures = [
        _case_change(baseline_by_id[case_id], candidate_by_id[case_id])
        for case_id in common_ids
        if baseline_by_id[case_id].passed and not candidate_by_id[case_id].passed
    ]
    fixed_cases = [
        _case_change(baseline_by_id[case_id], candidate_by_id[case_id])
        for case_id in common_ids
        if not baseline_by_id[case_id].passed and candidate_by_id[case_id].passed
    ]
    tag_regressions = _tag_regressions(baseline_by_id, candidate_by_id, common_ids)
    gate_failures = _gate_failures(metrics, gates, fail_if_drop)

    return CompareSummary(
        baseline_path=str(baseline_path),
        candidate_path=str(candidate_path),
        metrics=metrics,
        new_failures=new_failures,
        fixed_cases=fixed_cases,
        tag_regressions=tag_regressions,
        gate_triggered=bool(gate_failures),
        gate_failures=gate_failures,
    )


def _metric_changes(
    baseline: list[CaseRunRecord],
    candidate: list[CaseRunRecord],
) -> dict[str, MetricChange]:
    baseline_metrics = _run_metrics(baseline)
    candidate_metrics = _run_metrics(candidate)
    return {
        name: MetricChange(
            baseline=baseline_metrics.get(name, 0.0),
            candidate=candidate_metrics.get(name, 0.0),
            delta=candidate_metrics.get(name, 0.0) - baseline_metrics.get(name, 0.0),
        )
        for name in sorted(set(baseline_metrics) | set(candidate_metrics))
    }


def _run_metrics(records: list[CaseRunRecord]) -> dict[str, float]:
    count = len(records)
    if count == 0:
        return {
            "pass_rate": 0.0,
            "aggregate_score": 0.0,
            "task_success_rate": 0.0,
            "tool_call_accuracy": 0.0,
            "final_answer_correctness": 0.0,
            "safety_violation_rate": 0.0,
        }
    return {
        "pass_rate": sum(1 for record in records if record.passed) / count,
        "aggregate_score": sum(record.aggregate_score for record in records) / count,
        "task_success_rate": _average_score(records, "task_success"),
        "tool_call_accuracy": _average_score(records, "tool_call_accuracy"),
        "final_answer_correctness": _average_score(records, "final_answer_correctness"),
        "safety_violation_rate": 1.0 - _average_score(records, "safety"),
    }


def _average_score(records: list[CaseRunRecord], name: str) -> float:
    if not records:
        return 0.0
    return sum(record.scores.get(name, 0.0) for record in records) / len(records)


def _case_change(baseline: CaseRunRecord, candidate: CaseRunRecord) -> CaseChange:
    return CaseChange(
        case_id=candidate.case_id,
        baseline_failure_type=baseline.failure_type,
        candidate_failure_type=candidate.failure_type,
        baseline_score=baseline.aggregate_score,
        candidate_score=candidate.aggregate_score,
        candidate_reason=candidate.reason,
    )


def _tag_regressions(
    baseline_by_id: dict[str, CaseRunRecord],
    candidate_by_id: dict[str, CaseRunRecord],
    common_ids: list[str],
) -> list[TagRegression]:
    tags = sorted({tag for case_id in common_ids for tag in baseline_by_id[case_id].tags})
    regressions: list[TagRegression] = []
    for tag in tags:
        tagged_ids = [case_id for case_id in common_ids if tag in baseline_by_id[case_id].tags]
        if not tagged_ids:
            continue
        baseline_rate = sum(1 for case_id in tagged_ids if baseline_by_id[case_id].passed) / len(
            tagged_ids
        )
        candidate_rate = sum(
            1 for case_id in tagged_ids if candidate_by_id[case_id].passed
        ) / len(tagged_ids)
        delta = candidate_rate - baseline_rate
        if delta < 0:
            regressions.append(
                TagRegression(
                    tag=tag,
                    baseline_pass_rate=baseline_rate,
                    candidate_pass_rate=candidate_rate,
                    delta=delta,
                    case_count=len(tagged_ids),
                )
            )
    return regressions


def _load_gates(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    gates = payload.get("gates", {})
    if not isinstance(gates, dict):
        raise ValueError(f"gates config must contain a mapping at {path}")
    parsed: dict[str, dict[str, float]] = {}
    for metric, config in gates.items():
        if not isinstance(config, dict):
            raise ValueError(f"gate for {metric} must be a mapping")
        parsed[str(metric)] = {
            str(key): float(value)
            for key, value in config.items()
            if key in {"min", "max", "fail_if_drop_greater_than"}
        }
    return parsed


def _gate_failures(
    metrics: dict[str, MetricChange],
    gates: dict[str, dict[str, float]],
    fail_if_drop: dict[str, float],
) -> list[GateFailure]:
    combined = {metric: dict(config) for metric, config in gates.items()}
    for metric, threshold in fail_if_drop.items():
        combined.setdefault(metric, {})["fail_if_drop_greater_than"] = threshold

    failures: list[GateFailure] = []
    for metric, config in sorted(combined.items()):
        change = metrics.get(metric)
        if change is None:
            continue
        min_value = config.get("min")
        if min_value is not None and change.candidate < min_value:
            failures.append(
                GateFailure(
                    metric=metric,
                    baseline=change.baseline,
                    candidate=change.candidate,
                    threshold=min_value,
                    reason=f"{metric} {change.candidate:.4f} is below min {min_value:.4f}",
                )
            )
        max_value = config.get("max")
        if max_value is not None and change.candidate > max_value:
            failures.append(
                GateFailure(
                    metric=metric,
                    baseline=change.baseline,
                    candidate=change.candidate,
                    threshold=max_value,
                    reason=f"{metric} {change.candidate:.4f} is above max {max_value:.4f}",
                )
            )
        threshold = config.get("fail_if_drop_greater_than")
        drop = change.baseline - change.candidate
        if threshold is not None and drop > threshold:
            failures.append(
                GateFailure(
                    metric=metric,
                    baseline=change.baseline,
                    candidate=change.candidate,
                    drop=drop,
                    threshold=threshold,
                    reason=f"{metric} dropped {drop:.4f}, greater than {threshold:.4f}",
                )
            )
    return failures


def _write_json(path: Path, summary: CompareSummary) -> None:
    path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")


def _write_markdown(path: Path, summary: CompareSummary) -> None:
    lines = [
        "# Compare Report",
        "",
        "## Inputs",
        "",
        f"- Baseline: `{summary.baseline_path}`",
        f"- Candidate: `{summary.candidate_path}`",
        "",
        "## Metric Changes",
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, change in summary.metrics.items():
        lines.append(
            f"| `{metric}` | {change.baseline:.4f} | {change.candidate:.4f} | {change.delta:+.4f} |"
        )

    lines.extend(["", "## New Failures", ""])
    lines.extend(_case_table(summary.new_failures, empty="No new failures."))
    lines.extend(["", "## Fixed Cases", ""])
    lines.extend(_case_table(summary.fixed_cases, empty="No fixed cases."))
    lines.extend(["", "## Tag Regressions", ""])
    if summary.tag_regressions:
        lines.extend(["| Tag | Cases | Baseline pass rate | Candidate pass rate | Delta |", "| --- | ---: | ---: | ---: | ---: |"])
        for regression in summary.tag_regressions:
            lines.append(
                f"| `{regression.tag}` | {regression.case_count} | "
                f"{regression.baseline_pass_rate:.1%} | {regression.candidate_pass_rate:.1%} | "
                f"{regression.delta:+.1%} |"
            )
    else:
        lines.append("No tag regressions.")

    lines.extend(["", "## Gate", ""])
    if summary.gate_failures:
        lines.extend(["Gate triggered.", "", "| Metric | Reason |", "| --- | --- |"])
        for failure in summary.gate_failures:
            lines.append(f"| `{failure.metric}` | {_escape_table(failure.reason)} |")
    else:
        lines.append("Gate passed.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _case_table(changes: list[CaseChange], *, empty: str) -> list[str]:
    if not changes:
        return [empty]
    lines = ["| Case | Baseline | Candidate | Failure | Reason |", "| --- | ---: | ---: | --- | --- |"]
    for change in changes:
        lines.append(
            f"| `{change.case_id}` | {change.baseline_score:.3f} | {change.candidate_score:.3f} | "
            f"{change.candidate_failure_type} | {_escape_table(change.candidate_reason)} |"
        )
    return lines


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
