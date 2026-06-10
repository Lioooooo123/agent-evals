from __future__ import annotations

from agent_evals.scorers import ExecutionMetricsScorer, ScoringContext
from agent_evals.traces.schema import EvalCase, Trace, TraceMetrics, TraceStep


def _case(expected_extra: dict | None = None, metadata_extra: dict | None = None) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case_metrics",
            "input": {"messages": [{"role": "user", "content": "run"}]},
            "expected": expected_extra or {},
            "metadata": metadata_extra or {},
        }
    )


def _trace(*, latency_ms: int = 0, cost_usd: float = 0, steps: int = 0, status: str = "completed") -> Trace:
    return Trace(
        trace_id="trace_metrics",
        run_id="run_metrics",
        case_id="case_metrics",
        agent_version="test",
        status=status,
        metrics=TraceMetrics(latency_ms=latency_ms, cost_usd=cost_usd),
        steps=[
            TraceStep(step_id=f"step_{index}", index=index, type="llm")
            for index in range(1, steps + 1)
        ],
    )


def test_execution_metrics_skips_unconfigured_limits():
    result = ExecutionMetricsScorer().score(
        _case(),
        _trace(latency_ms=999_999, cost_usd=99, steps=99),
        ScoringContext(),
    )

    assert result.passed
    assert result.metadata["checks"] == {}


def test_execution_metrics_checks_configured_limits():
    result = ExecutionMetricsScorer().score(
        _case(
            {
                "max_latency_ms": 1000,
                "max_cost_usd": 0.25,
                "max_steps": 3,
                "timeout_s": 2,
            }
        ),
        _trace(latency_ms=2500, cost_usd=0.50, steps=4),
        ScoringContext(),
    )

    assert not result.passed
    assert "latency_ms" in result.reason
    assert "cost_usd" in result.reason
    assert "step count" in result.reason
    assert "timeout" in result.reason
