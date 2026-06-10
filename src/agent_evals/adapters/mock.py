"""Mock adapter for deterministic demos and tests."""

from __future__ import annotations

from typing import Any

from agent_evals.adapters.base import AgentOutput
from agent_evals.traces.schema import EvalCase, Trace, TraceMetrics, TraceStep, TraceToolCall


class MockAgentAdapter:
    name = "mock"

    def run(self, case: EvalCase, run_id: str) -> AgentOutput:
        mock_config = _mock_config(case)
        script = str(mock_config.get("script", "correct"))
        latency_ms = int(mock_config.get("latency_ms", 100))
        cost_usd = float(mock_config.get("cost_usd", 0.0))

        tool_specs = _tool_specs(case, script, mock_config)
        steps: list[TraceStep] = []
        for index, tool_spec in enumerate(tool_specs, 1):
            tool_name = str(tool_spec["tool_name"])
            arguments = dict(tool_spec.get("arguments", {}))
            tool_call_id = f"{case.id}_tool_{index}"
            steps.append(
                TraceStep(
                    step_id=f"{case.id}_step_{len(steps) + 1}",
                    index=len(steps) + 1,
                    type="tool_call",
                    summary=f"Mock called {tool_name}",
                    tool_call=TraceToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                        tool_call_id=tool_call_id,
                    ),
                    metrics=TraceMetrics(latency_ms=latency_ms // max(len(tool_specs), 1)),
                )
            )
            steps.append(
                TraceStep(
                    step_id=f"{case.id}_step_{len(steps) + 1}",
                    index=len(steps) + 1,
                    type="observation",
                    summary=f"Mock observed {tool_name}",
                    observation={
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "content": tool_spec.get("observation", {"ok": True}),
                        "is_error": False,
                    },
                )
            )

        final_answer = _final_answer(case, script, mock_config)
        if final_answer:
            steps.append(
                TraceStep(
                    step_id=f"{case.id}_step_{len(steps) + 1}",
                    index=len(steps) + 1,
                    type="llm",
                    summary=final_answer,
                )
            )

        trace = Trace(
            trace_id=f"trace_{case.id}",
            run_id=run_id,
            case_id=case.id,
            agent_version="mock-agent-v0",
            final_answer=final_answer,
            metrics=TraceMetrics(latency_ms=latency_ms, cost_usd=cost_usd),
            steps=steps,
        )
        return AgentOutput(trace=trace)


def _mock_config(case: EvalCase) -> dict[str, Any]:
    metadata_extra = case.metadata.model_extra or {}
    mock_config = metadata_extra.get("mock", {})
    return mock_config if isinstance(mock_config, dict) else {}


def _tool_specs(
    case: EvalCase,
    script: str,
    mock_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if isinstance(mock_config.get("tool_calls"), list):
        return [dict(call) for call in mock_config["tool_calls"]]

    specs = [
        {
            "tool_name": expected.tool_name,
            "arguments": dict(expected.arguments),
            "observation": {"ok": True, "tool": expected.tool_name},
        }
        for expected in case.expected.tool_calls
    ]
    if script == "wrong_tool_args" and specs:
        specs[0]["arguments"] = _wrong_arguments(specs[0]["arguments"])
    return specs


def _wrong_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    wrong = dict(arguments)
    if not wrong:
        wrong["unexpected"] = "wrong"
        return wrong
    first_key = next(iter(wrong))
    wrong[first_key] = f"{wrong[first_key]}_wrong"
    return wrong


def _final_answer(
    case: EvalCase,
    script: str,
    mock_config: dict[str, Any],
) -> str:
    configured = mock_config.get("final_answer")
    if isinstance(configured, str):
        return configured
    if script == "missing_keyword":
        return "The mock agent completed the task."
    if case.expected.answer_contains:
        return " ".join(case.expected.answer_contains) + " completed by mock agent."
    return "Mock agent completed the task."
