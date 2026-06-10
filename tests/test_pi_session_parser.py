from __future__ import annotations

from pathlib import Path

from agent_evals.parsers.pi_session import parse_pi_session_jsonl


FIXTURE = Path(__file__).resolve().parents[1] / "cases" / "fixtures" / "pi_session_sample.jsonl"


def test_parse_pi_session_fixture_has_steps():
    trace = parse_pi_session_jsonl(FIXTURE)

    assert trace.steps
    assert trace.metrics.total_tokens > 0


def test_parse_pi_session_fixture_bash_mapping_counts():
    trace = parse_pi_session_jsonl(FIXTURE)

    model_bash_calls = [
        step
        for step in trace.steps
        if step.type == "tool_call"
        and step.origin == "model"
        and step.tool_call is not None
        and step.tool_call.tool_name == "bash"
    ]
    observations_by_tool_call_id = {
        step.observation.get("tool_call_id"): step
        for step in trace.steps
        if step.type == "observation"
        and step.observation
        and step.observation.get("tool_call_id")
    }

    assert len(model_bash_calls) == 97
    for step in model_bash_calls:
        assert step.tool_call is not None
        assert step.tool_call.tool_call_id in observations_by_tool_call_id


def test_parse_pi_session_fixture_bash_execution_is_non_model():
    trace = parse_pi_session_jsonl(FIXTURE)

    non_model_steps = [step for step in trace.steps if step.origin == "non_model"]
    non_model_bash_steps = [
        step
        for step in non_model_steps
        if step.origin == "non_model"
        and step.tool_call is not None
        and step.tool_call.tool_name == "bash"
    ]

    assert len(non_model_steps) == 1
    assert len(non_model_bash_steps) == 1
    assert non_model_bash_steps[0].observation is not None
    assert non_model_bash_steps[0].observation["exit_code"] == 0
    assert non_model_bash_steps[0].tool_call.arguments["command"].startswith("rm ")
