from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evals.scorers import AggregateScorer, ScoreResult, ScoringContext
from agent_evals.scorers.aggregate import WeightConfig
from agent_evals.traces.schema import EvalCase, Trace


def _case() -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case_aggregate",
            "input": {"messages": [{"role": "user", "content": "score"}]},
        }
    )


def _trace() -> Trace:
    return Trace(
        trace_id="trace_aggregate",
        run_id="run_aggregate",
        case_id="case_aggregate",
        agent_version="test",
    )


def _score(name: str, score: float) -> ScoreResult:
    return ScoreResult(
        name=name,
        score=score,
        passed=score > 0,
        reason="test score",
    )


def test_aggregate_hard_fail_overrides_high_total_score():
    scorer = AggregateScorer(
        WeightConfig(
            weights={"task_success": 0.5, "safety": 0.5},
            pass_threshold=0.8,
            hard_fail={"safety": 0},
        )
    )

    result = scorer.score(
        _case(),
        _trace(),
        ScoringContext(
            score_results=[
                _score("task_success", 1.0),
                _score("safety", 0.0),
            ]
        ),
    )

    assert result.score == 0.5
    assert not result.passed
    assert result.metadata["hard_fail_hits"] == {"safety": 0.0}


def test_aggregate_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValidationError, match="must sum to 1.0"):
        WeightConfig(
            weights={"task_success": 0.7, "safety": 0.7},
            pass_threshold=0.8,
        )


def test_aggregate_loads_default_weights_file():
    scorer = AggregateScorer.from_yaml("configs/weights.yaml")

    assert scorer.config.pass_threshold == 0.8
    assert sum(scorer.config.weights.values()) == pytest.approx(1.0)
