"""End-to-end run pipeline."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.adapters import AgentAdapter, MockAgentAdapter, PiAgentAdapter
from agent_evals.datasets import load_eval_cases_jsonl
from agent_evals.reports.files import write_eval_report, write_failed_cases, write_summary_csv
from agent_evals.scorers import (
    AggregateScorer,
    AnswerRuleScorer,
    CommandPassScorer,
    ExecutionMetricsScorer,
    FinalAnswerGroundingScorer,
    NoUncommittedNoiseScorer,
    ScoreResult,
    ScoringContext,
    TaskSuccessJudgeScorer,
    ToolTrajectoryScorer,
    WorkspaceDiffScorer,
    judge_dimension_results,
)
from agent_evals.traces.schema import EvalCase, Trace


class CaseRunRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    case_id: str
    passed: bool = Field(alias="pass")
    aggregate_score: float
    scores: dict[str, float]
    failure_type: str
    reason: str
    trace_path: str
    latency_ms: int
    cost_usd: float
    tags: list[str] = Field(default_factory=list)
    score_results: list[ScoreResult] = Field(default_factory=list)


class RunOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cases_path: Path
    out_dir: Path
    adapter_name: str = "mock"
    weights_path: Path = Path("configs/weights.yaml")
    rubric_path: Path | None = Path("configs/judge_rubric.yaml")
    judge_client: Any = None
    run_id: str | None = None


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    out_dir: Path
    records: list[CaseRunRecord]


def run_eval(options: RunOptions) -> RunSummary:
    run_id = options.run_id or _default_run_id()
    run_dir = options.out_dir / run_id
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    cases = load_eval_cases_jsonl(options.cases_path)
    adapter = _adapter(options.adapter_name)
    aggregate_scorer = AggregateScorer.from_yaml(options.weights_path)
    judge_scorer = (
        TaskSuccessJudgeScorer.from_yaml(options.rubric_path, client=options.judge_client)
        if options.rubric_path is not None
        else None
    )
    deterministic_scorers = [
        AnswerRuleScorer(),
        ToolTrajectoryScorer(),
        ExecutionMetricsScorer(),
    ]
    if options.adapter_name == "pi":
        deterministic_scorers.extend(
            [
                WorkspaceDiffScorer(),
                CommandPassScorer(),
                FinalAnswerGroundingScorer(),
                NoUncommittedNoiseScorer(),
            ]
        )

    records: list[CaseRunRecord] = []
    for case in cases:
        output = adapter.run(case, run_id)
        trace = output.trace
        trace_path = traces_dir / f"{case.id}.json"
        trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")

        score_results: list[ScoreResult] = []
        context = ScoringContext(score_results=score_results)
        for scorer in deterministic_scorers:
            score_results.append(scorer.score(case, trace, context))
        if judge_scorer is not None:
            judge_result = judge_scorer.score(case, trace, context)
            score_results.append(judge_result)
            score_results.extend(judge_dimension_results(judge_result))
        score_results.extend(_placeholder_scores(case, trace, score_results))

        aggregate = aggregate_scorer.score(
            case,
            trace,
            ScoringContext(score_results=score_results),
        )
        all_results = [*score_results, aggregate]
        records.append(_record(case, trace, trace_path, aggregate, all_results))

    _write_eval_results(run_dir / "eval_results.jsonl", records)
    write_summary_csv(run_dir / "summary.csv", records)
    write_eval_report(run_dir / "eval_report.md", run_id, records)
    write_failed_cases(run_dir / "failed_cases.md", cases, records, traces_dir)
    return RunSummary(run_id=run_id, out_dir=run_dir, records=records)


def _adapter(name: str) -> AgentAdapter:
    if name == "mock":
        return MockAgentAdapter()
    if name == "pi":
        return PiAgentAdapter()
    raise ValueError(f"unsupported adapter: {name}")


def _placeholder_scores(
    case: EvalCase,
    trace: Trace,
    existing: list[ScoreResult],
) -> list[ScoreResult]:
    existing_names = {result.name for result in existing}
    placeholders: list[ScoreResult] = []

    if "task_success" not in existing_names:
        expected_success = (
            case.expected.outcome.task_success
            if case.expected.outcome is not None and case.expected.outcome.task_success is not None
            else True
        )
        success = trace.status == "completed" if expected_success else trace.status != "completed"
        placeholders.append(
            ScoreResult(
                name="task_success",
                score=1.0 if success else 0.0,
                passed=success,
                reason="deterministic task success placeholder until judge is implemented",
                failure_type="none" if success else "incomplete_task",
                metadata={"source": "placeholder_until_judge"},
            )
        )

    for name in ("grounding", "safety", "format_compliance"):
        if name not in existing_names:
            placeholders.append(
                ScoreResult(
                    name=name,
                    score=1.0,
                    passed=True,
                    reason=f"{name} placeholder until judge/rule scorer is implemented",
                    metadata={"source": "placeholder_until_judge"},
                )
            )
    return placeholders


def _record(
    case: EvalCase,
    trace: Trace,
    trace_path: Path,
    aggregate: ScoreResult,
    score_results: list[ScoreResult],
) -> CaseRunRecord:
    scores = {result.name: result.score for result in score_results}
    failed_results = [result for result in score_results if not result.passed]
    primary_failure = next(
        (result for result in failed_results if result.name != "aggregate"),
        aggregate if not aggregate.passed else None,
    )
    return CaseRunRecord(
        case_id=case.id,
        passed=aggregate.passed,
        aggregate_score=aggregate.score,
        scores=scores,
        failure_type=primary_failure.failure_type if primary_failure else "none",
        reason=primary_failure.reason if primary_failure else aggregate.reason,
        trace_path=str(trace_path),
        latency_ms=trace.metrics.latency_ms,
        cost_usd=trace.metrics.cost_usd,
        tags=case.metadata.tags,
        score_results=score_results,
    )


def _write_eval_results(path: Path, records: list[CaseRunRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json(by_alias=True))
            handle.write("\n")


def summarize_records(records: list[CaseRunRecord]) -> dict[str, Any]:
    case_count = len(records)
    pass_count = sum(1 for record in records if record.passed)
    failure_counts = Counter(record.failure_type for record in records if not record.passed)
    tag_totals: dict[str, list[bool]] = defaultdict(list)
    for record in records:
        for tag in record.tags:
            tag_totals[tag].append(record.passed)
    return {
        "case_count": case_count,
        "pass_count": pass_count,
        "pass_rate": pass_count / case_count if case_count else 0.0,
        "avg_latency_ms": sum(record.latency_ms for record in records) / case_count
        if case_count
        else 0.0,
        "avg_cost_usd": sum(record.cost_usd for record in records) / case_count
        if case_count
        else 0.0,
        "failure_counts": dict(failure_counts),
        "tag_slices": {
            tag: {
                "case_count": len(values),
                "pass_count": sum(1 for passed in values if passed),
                "pass_rate": sum(1 for passed in values if passed) / len(values),
            }
            for tag, values in sorted(tag_totals.items())
        },
    }


def _default_run_id() -> str:
    return datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
