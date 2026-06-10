"""LLM-as-a-judge scorer for task success."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent_evals.scorers.base import ScoreResult, ScoringContext
from agent_evals.traces.schema import EvalCase, Trace, TraceStep


class JudgeSkippedError(RuntimeError):
    """Raised when judge scoring is configured but cannot run."""


class JudgeProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "anthropic"
    model: str
    api_key_env: str = "ANTHROPIC_API_KEY"


class JudgeRubricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    provider: JudgeProviderConfig
    rubric: dict[str, str]
    instructions: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "JudgeRubricConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls.model_validate(payload)


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_completion: float = Field(ge=0.0, le=1.0)
    tool_use: float = Field(ge=0.0, le=1.0)
    grounding: float = Field(ge=0.0, le=1.0)
    efficiency: float = Field(ge=0.0, le=1.0)
    safety: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool = Field(alias="pass")
    failure_type: str
    reason: str


class JudgeClient(Protocol):
    def complete(self, prompt: str, config: JudgeRubricConfig) -> str:
        ...


class AnthropicJudgeClient:
    """Minimal Anthropic Messages API client using the standard library."""

    api_url = "https://api.anthropic.com/v1/messages"

    def complete(self, prompt: str, config: JudgeRubricConfig) -> str:
        api_key = os.environ.get(config.provider.api_key_env)
        if not api_key:
            raise JudgeSkippedError(
                f"missing API key env var {config.provider.api_key_env}"
            )
        if config.provider.name != "anthropic":
            raise JudgeSkippedError(f"unsupported judge provider {config.provider.name!r}")

        payload = {
            "model": config.provider.model,
            "max_tokens": 1200,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"judge provider request failed: {exc}") from exc

        content = data.get("content", [])
        text_blocks = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_blocks)


class TaskSuccessJudgeScorer:
    name = "task_success"

    def __init__(
        self,
        config: JudgeRubricConfig,
        client: JudgeClient | None = None,
    ):
        self.config = config
        self.client = client or AnthropicJudgeClient()

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        client: JudgeClient | None = None,
    ) -> "TaskSuccessJudgeScorer":
        return cls(JudgeRubricConfig.from_yaml(path), client=client)

    def score(
        self,
        case: EvalCase,
        trace: Trace,
        context: ScoringContext,
    ) -> ScoreResult:
        prompt = _build_prompt(case, trace, self.config)
        try:
            first_response = self.client.complete(prompt, self.config)
        except JudgeSkippedError as exc:
            return _skipped_result(self.config, str(exc))

        first_error: str | None = None
        for attempt, response in enumerate([first_response, None], 1):
            if response is None:
                try:
                    response = self.client.complete(
                        prompt + "\n\nPrevious response was invalid JSON. Return JSON only.",
                        self.config,
                    )
                except JudgeSkippedError as exc:
                    return _skipped_result(self.config, str(exc))
            try:
                output = JudgeOutput.model_validate(json.loads(response))
            except (json.JSONDecodeError, ValidationError) as exc:
                first_error = str(exc)
                if attempt == 1:
                    continue
                return ScoreResult(
                    name=self.name,
                    score=0.0,
                    passed=False,
                    reason=f"judge output parse failed after retry: {first_error}",
                    failure_type="judge_error",
                    metadata={
                        "rubric_version": self.config.version,
                        "provider": self.config.provider.name,
                        "model": self.config.provider.model,
                        "attempts": 2,
                    },
                )
            return _score_result_from_output(output, self.config, attempt)

        raise AssertionError("unreachable judge retry loop")


def judge_dimension_results(judge_result: ScoreResult) -> list[ScoreResult]:
    if judge_result.metadata.get("status") == "skipped":
        return []
    dimensions = judge_result.metadata.get("dimensions")
    if not isinstance(dimensions, dict):
        return []
    results: list[ScoreResult] = []
    for name in ("grounding", "safety"):
        if name in dimensions:
            score = float(dimensions[name])
            results.append(
                ScoreResult(
                    name=name,
                    score=score,
                    passed=score > 0,
                    reason=f"judge {name} score",
                    failure_type="none" if score > 0 else "judge_error",
                    metadata={
                        "source": "judge",
                        "rubric_version": judge_result.metadata.get("rubric_version"),
                    },
                )
            )
    return results


def _score_result_from_output(
    output: JudgeOutput,
    config: JudgeRubricConfig,
    attempts: int,
) -> ScoreResult:
    return ScoreResult(
        name="task_success",
        score=output.overall_score,
        passed=output.passed,
        reason=output.reason,
        failure_type=output.failure_type,
        metadata={
            "source": "judge",
            "rubric_version": config.version,
            "provider": config.provider.name,
            "model": config.provider.model,
            "attempts": attempts,
            "dimensions": {
                "goal_completion": output.goal_completion,
                "tool_use": output.tool_use,
                "grounding": output.grounding,
                "efficiency": output.efficiency,
                "safety": output.safety,
            },
        },
    )


def _skipped_result(config: JudgeRubricConfig, reason: str) -> ScoreResult:
    return ScoreResult(
        name="task_success_judge",
        score=0.0,
        passed=True,
        reason=f"judge skipped: {reason}",
        metadata={
            "status": "skipped",
            "rubric_version": config.version,
            "provider": config.provider.name,
            "model": config.provider.model,
        },
    )


def _build_prompt(case: EvalCase, trace: Trace, config: JudgeRubricConfig) -> str:
    payload = {
        "user_task": [message.model_dump(mode="json") for message in case.input.messages],
        "expected": case.expected.model_dump(mode="json"),
        "final_answer": trace.final_answer,
        "trace_summary": _trace_summary(trace.steps),
        "rubric": config.rubric,
        "instructions": config.instructions,
        "required_json_schema": {
            "goal_completion": "number 0..1",
            "tool_use": "number 0..1",
            "grounding": "number 0..1",
            "efficiency": "number 0..1",
            "safety": "number 0..1",
            "overall_score": "number 0..1",
            "pass": "boolean",
            "failure_type": "string",
            "reason": "string",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _trace_summary(steps: list[TraceStep]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for step in steps:
        row: dict[str, Any] = {
            "index": step.index,
            "type": step.type,
            "origin": step.origin,
            "summary": step.summary,
        }
        if step.tool_call is not None:
            row["tool_call"] = step.tool_call.model_dump(mode="json")
        if step.observation is not None:
            row["observation"] = step.observation
        summary.append(row)
    return summary
