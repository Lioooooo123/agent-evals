"""Execution metric limit checks."""

from __future__ import annotations

from typing import Any

from agent_evals.scorers.base import ScoreResult, ScoringContext
from agent_evals.traces.schema import EvalCase, Trace


class ExecutionMetricsScorer:
    name = "efficiency"

    def score(
        self,
        case: EvalCase,
        trace: Trace,
        context: ScoringContext,
    ) -> ScoreResult:
        failures: list[str] = []
        checks: dict[str, Any] = {}

        max_latency_ms = _limit(case, "max_latency_ms")
        if max_latency_ms is not None:
            checks["max_latency_ms"] = max_latency_ms
            if trace.metrics.latency_ms > max_latency_ms:
                failures.append(
                    f"latency_ms {trace.metrics.latency_ms} exceeds {max_latency_ms}"
                )

        max_cost_usd = _limit(case, "max_cost_usd")
        if max_cost_usd is not None:
            checks["max_cost_usd"] = max_cost_usd
            if trace.metrics.cost_usd > max_cost_usd:
                failures.append(f"cost_usd {trace.metrics.cost_usd} exceeds {max_cost_usd}")

        max_steps = _limit(case, "max_steps")
        if max_steps is not None:
            checks["max_steps"] = max_steps
            if len(trace.steps) > max_steps:
                failures.append(f"step count {len(trace.steps)} exceeds {max_steps}")

        timeout_s = _limit(case, "timeout_s") or _limit(case, "max_timeout_s")
        if timeout_s is not None:
            checks["timeout_s"] = timeout_s
            if trace.status == "timeout":
                failures.append("trace status is timeout")
            if trace.metrics.latency_ms > timeout_s * 1000:
                failures.append(
                    f"latency_ms {trace.metrics.latency_ms} exceeds timeout {timeout_s}s"
                )

        passed = not failures
        return ScoreResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="execution metrics passed" if passed else "; ".join(failures),
            failure_type="none" if passed else "inefficient",
            metadata={"checks": checks},
        )


def _limit(case: EvalCase, key: str) -> Any:
    expected_extra = case.expected.model_extra or {}
    metadata_extra = case.metadata.model_extra or {}

    for container in (
        expected_extra,
        metadata_extra,
        expected_extra.get("metrics") if isinstance(expected_extra.get("metrics"), dict) else {},
        metadata_extra.get("metrics") if isinstance(metadata_extra.get("metrics"), dict) else {},
    ):
        if key in container:
            return container[key]
    return None
