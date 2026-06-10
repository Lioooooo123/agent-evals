"""Adapter interface for running agents under evaluation."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from agent_evals.traces.schema import EvalCase, Trace


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace: Trace


class AgentAdapter(Protocol):
    name: str

    def run(self, case: EvalCase, run_id: str) -> AgentOutput:
        ...
