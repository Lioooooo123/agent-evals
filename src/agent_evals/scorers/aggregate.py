"""Weighted aggregate scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_evals.scorers.base import ScoreResult, ScoringContext
from agent_evals.traces.schema import EvalCase, Trace


class WeightConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float]
    pass_threshold: float
    hard_fail: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_weights(self) -> "WeightConfig":
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"aggregate weights must sum to 1.0, got {total}")
        return self


class AggregateScorer:
    name = "aggregate"

    def __init__(self, config: WeightConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AggregateScorer":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(WeightConfig.model_validate(payload))

    def score(
        self,
        case: EvalCase,
        trace: Trace,
        context: ScoringContext,
    ) -> ScoreResult:
        results_by_name = {result.name: result for result in context.score_results}
        missing = [
            name for name in self.config.weights if name not in results_by_name
        ]
        if missing:
            return ScoreResult(
                name=self.name,
                score=0.0,
                passed=False,
                reason=f"missing required score results: {missing}",
                failure_type="runtime_error",
                metadata={"missing": missing},
            )

        weighted_score = sum(
            self.config.weights[name] * results_by_name[name].score
            for name in self.config.weights
        )
        hard_fail_hits = {
            name: threshold
            for name, threshold in self.config.hard_fail.items()
            if name in results_by_name and results_by_name[name].score <= threshold
        }
        passed = weighted_score >= self.config.pass_threshold and not hard_fail_hits
        reason = (
            "aggregate score passed"
            if passed
            else _failure_reason(weighted_score, self.config.pass_threshold, hard_fail_hits)
        )
        return ScoreResult(
            name=self.name,
            score=weighted_score,
            passed=passed,
            reason=reason,
            failure_type="none" if passed else "incomplete_task",
            metadata={
                "weights": self.config.weights,
                "pass_threshold": self.config.pass_threshold,
                "hard_fail_hits": hard_fail_hits,
            },
        )


def _failure_reason(
    score: float,
    threshold: float,
    hard_fail_hits: dict[str, Any],
) -> str:
    reasons: list[str] = []
    if score < threshold:
        reasons.append(f"score {score:.3f} below threshold {threshold:.3f}")
    if hard_fail_hits:
        reasons.append(f"hard_fail triggered: {sorted(hard_fail_hits)}")
    return "; ".join(reasons)
