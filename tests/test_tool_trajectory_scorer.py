from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals.parsers.pi_session import parse_pi_session_jsonl
from agent_evals.scorers import ScoringContext, ToolTrajectoryScorer
from agent_evals.traces.schema import EvalCase, Trace, TraceStep, TraceToolCall


FIXTURE = Path(__file__).resolve().parents[1] / "cases" / "fixtures" / "pi_session_sample.jsonl"


def _case(tool_calls: list[dict]) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case_tools",
            "input": {"messages": [{"role": "user", "content": "use tools"}]},
            "expected": {"tool_calls": tool_calls},
        }
    )


def _trace(calls: list[tuple[str, dict]], *, include_non_model: bool = False) -> Trace:
    steps: list[TraceStep] = []
    for index, (tool_name, arguments) in enumerate(calls, 1):
        steps.append(
            TraceStep(
                step_id=f"step_{index}",
                index=index,
                type="tool_call",
                summary=f"{tool_name} call",
                tool_call=TraceToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    tool_call_id=f"call_{index}",
                ),
            )
        )
    if include_non_model:
        steps.append(
            TraceStep(
                step_id="step_non_model",
                index=len(steps) + 1,
                type="tool_call",
                origin="non_model",
                summary="non-model bash",
                tool_call=TraceToolCall(
                    tool_name="bash",
                    arguments={"command": "rm temp"},
                ),
            )
        )
    return Trace(
        trace_id="trace_tools",
        run_id="run_tools",
        case_id="case_tools",
        agent_version="test",
        steps=steps,
    )


@pytest.mark.parametrize(
    ("mode", "expected", "actual"),
    [
        (
            "strict",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "strict"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "strict"},
            ],
            [("read", {"path": "a"}), ("bash", {"command": "pytest"})],
        ),
        (
            "unordered",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "unordered"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "unordered"},
            ],
            [("bash", {"command": "pytest"}), ("read", {"path": "a"})],
        ),
        (
            "subset",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "subset"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "subset"},
            ],
            [
                ("read", {"path": "a"}),
                ("edit", {"path": "a", "old": "x", "new": "y"}),
                ("bash", {"command": "pytest", "cwd": "."}),
            ],
        ),
        (
            "superset",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "superset"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "superset"},
            ],
            [("read", {"path": "a"})],
        ),
    ],
)
def test_tool_trajectory_match_modes_pass(mode, expected, actual):
    result = ToolTrajectoryScorer().score(
        _case(expected),
        _trace(actual),
        ScoringContext(),
    )

    assert result.passed, mode
    assert result.score == 1.0


@pytest.mark.parametrize(
    ("mode", "expected", "actual"),
    [
        (
            "strict",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "strict"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "strict"},
            ],
            [("bash", {"command": "pytest"}), ("read", {"path": "a"})],
        ),
        (
            "unordered",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "unordered"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "unordered"},
            ],
            [("bash", {"command": "pytest"}), ("edit", {"path": "a"})],
        ),
        (
            "subset",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "subset"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "subset"},
            ],
            [("bash", {"command": "pytest"}), ("read", {"path": "a"})],
        ),
        (
            "superset",
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "superset"},
            ],
            [("read", {"path": "a"}), ("bash", {"command": "pytest"})],
        ),
    ],
)
def test_tool_trajectory_match_modes_fail(mode, expected, actual):
    result = ToolTrajectoryScorer().score(
        _case(expected),
        _trace(actual),
        ScoringContext(),
    )

    assert not result.passed, mode
    assert result.score == 0.0


def test_tool_trajectory_diagnoses_tool_selection_failure():
    result = ToolTrajectoryScorer().score(
        _case(
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "strict"},
            ]
        ),
        _trace([("bash", {"command": "pytest"})]),
        ScoringContext(),
    )

    assert result.failure_type == "tool_selection"
    assert "missing expected tool" in result.reason


def test_tool_trajectory_diagnoses_tool_arguments_failure():
    result = ToolTrajectoryScorer().score(
        _case(
            [
                {
                    "tool_name": "lookup_order",
                    "arguments": {"order_id": "A123"},
                    "match_mode": "strict",
                },
            ]
        ),
        _trace([("lookup_order", {"order_id": "A123_wrong"})]),
        ScoringContext(),
    )

    assert result.failure_type == "tool_arguments"
    assert "expected order_id=A123, got A123_wrong" in result.reason


def test_tool_trajectory_diagnoses_tool_order_failure():
    result = ToolTrajectoryScorer().score(
        _case(
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "strict"},
                {"tool_name": "bash", "arguments": {"command": "pytest"}, "match_mode": "strict"},
            ]
        ),
        _trace([("bash", {"command": "pytest"}), ("read", {"path": "a"})]),
        ScoringContext(),
    )

    assert result.failure_type == "tool_order"
    assert "wrong order" in result.reason


def test_tool_trajectory_supports_argument_subset_matching():
    result = ToolTrajectoryScorer().score(
        _case(
            [
                {
                    "tool_name": "bash",
                    "arguments": {"command": "pytest"},
                    "match_mode": "strict",
                    "argument_match_mode": "subset",
                }
            ]
        ),
        _trace([("bash", {"command": "pytest", "cwd": "."})]),
        ScoringContext(),
    )

    assert result.passed


def test_tool_trajectory_excludes_non_model_steps():
    result = ToolTrajectoryScorer().score(
        _case(
            [
                {"tool_name": "read", "arguments": {"path": "a"}, "match_mode": "strict"},
            ]
        ),
        _trace([("read", {"path": "a"})], include_non_model=True),
        ScoringContext(),
    )

    assert result.passed


def test_tool_trajectory_excludes_non_model_step_in_real_fixture_trace():
    trace = parse_pi_session_jsonl(FIXTURE)
    expected = [
        {
            "tool_name": step.tool_call.tool_name,
            "arguments": step.tool_call.arguments,
            "match_mode": "strict",
        }
        for step in trace.steps
        if step.type == "tool_call"
        and step.origin == "model"
        and step.tool_call is not None
    ]

    result = ToolTrajectoryScorer().score(
        _case(expected),
        trace,
        ScoringContext(),
    )

    assert result.passed
    assert result.metadata["actual_count"] == len(expected)
    assert len([step for step in trace.steps if step.origin == "non_model"]) == 1
