"""Pydantic data models for the Agent Evals spec."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


JsonDict = dict[str, Any]


class Message(BaseModel):
    """OpenAI/LangChain-compatible message shape."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any


class EvalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message]


class ExpectedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: JsonDict = Field(default_factory=dict)
    match_mode: Literal["exact", "strict", "unordered", "subset", "superset"] = "exact"
    argument_match_mode: Literal["exact", "subset"] = "exact"


class ExpectedWorkspace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files_changed: list[str] = Field(default_factory=list)
    files_forbidden: list[str] = Field(default_factory=list)
    allow_extra_changes: bool = True


class ExpectedCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd: str
    cwd: str
    timeout_s: int
    must_pass: bool


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_success: bool | None = None
    handoff_required: bool | None = None


class EvalExpected(BaseModel):
    """Expected outputs and outcome checks for a case."""

    model_config = ConfigDict(extra="allow")

    answer_contains: list[str] = Field(default_factory=list)
    answer_must_not_contain: list[str] = Field(default_factory=list)
    answer_regex: list[str] = Field(default_factory=list)
    answer_json_schema: JsonDict | None = None
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    workspace: ExpectedWorkspace | None = None
    commands: list[ExpectedCommand] = Field(default_factory=list)
    outcome: ExpectedOutcome | None = None


class EvalMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str | None = None
    difficulty: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    input: EvalInput
    expected: EvalExpected = Field(default_factory=EvalExpected)
    metadata: EvalMetadata = Field(default_factory=EvalMetadata)


class TraceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


class TraceToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: JsonDict = Field(default_factory=dict)
    tool_call_id: str | None = None


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    index: int
    type: Literal[
        "llm",
        "tool_call",
        "retriever",
        "observation",
        "retry",
        "handoff",
        "final",
        "error",
    ]
    timestamp: datetime | None = None
    summary: str = ""
    origin: Literal["model", "non_model"] = "model"
    tool_call: TraceToolCall | None = None
    observation: JsonDict | None = None
    error: str | None = None
    metrics: TraceMetrics = Field(default_factory=TraceMetrics)


class Trace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    run_id: str
    case_id: str
    agent_version: str
    status: Literal["completed", "failed", "timeout", "cancelled"] = "completed"
    started_at: datetime | None = None
    ended_at: datetime | None = None
    final_answer: str | None = None
    metrics: TraceMetrics = Field(default_factory=TraceMetrics)
    steps: list[TraceStep] = Field(default_factory=list)


class EvalResult(BaseModel):
    case_id: str
    run_id: str
    passed: bool = Field(alias="pass")
    scores: dict[str, float] = Field(default_factory=dict)
    failure_type: str = "none"
    reason: str = ""
    trace_path: str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
