"""Shared scorer interfaces and result models."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.traces.schema import EvalCase, Trace


class ScoreResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str
    score: float
    passed: bool = Field(alias="pass")
    reason: str = ""
    failure_type: str = "none"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoringContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    score_results: list[ScoreResult] = Field(default_factory=list)


class Scorer(Protocol):
    name: str

    def score(
        self,
        case: EvalCase,
        trace: Trace,
        context: ScoringContext,
    ) -> ScoreResult:
        ...
