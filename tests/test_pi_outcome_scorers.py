from __future__ import annotations

from agent_evals.adapters.pi import PI_COMMANDS_KIND, PI_WORKSPACE_KIND
from agent_evals.scorers import (
    CommandPassScorer,
    FinalAnswerGroundingScorer,
    NoUncommittedNoiseScorer,
    ScoringContext,
    WorkspaceDiffScorer,
)
from agent_evals.traces.schema import EvalCase, Trace, TraceStep


def test_workspace_diff_scorer_checks_expected_forbidden_and_extra_files():
    case = _case(
        {
            "workspace": {
                "files_changed": ["src/app.py"],
                "files_forbidden": [".env"],
                "allow_extra_changes": False,
            }
        }
    )
    trace = _trace(changed_files=["src/app.py", ".env", "tmp.log"])

    result = WorkspaceDiffScorer().score(case, trace, ScoringContext())

    assert not result.passed
    assert result.failure_type == "workspace_diff"
    assert "forbidden files changed" in result.reason
    assert "unexpected files changed" in result.reason


def test_command_pass_scorer_fails_required_failed_commands():
    case = _case(
        {
            "commands": [
                {"cmd": "pytest", "cwd": ".", "timeout_s": 30, "must_pass": True}
            ]
        }
    )
    trace = _trace(command_results=[{"cmd": "pytest", "exit_code": 1, "must_pass": True}])

    result = CommandPassScorer().score(case, trace, ScoringContext())

    assert not result.passed
    assert result.failure_type == "command_failed"
    assert "pytest" in result.reason


def test_final_answer_grounding_flags_success_claim_after_failed_commands():
    case = _case()
    trace = _trace(
        final_answer="Fixed and tests passed.",
        command_results=[{"cmd": "pytest", "exit_code": 1, "must_pass": True}],
    )

    result = FinalAnswerGroundingScorer().score(case, trace, ScoringContext())

    assert not result.passed
    assert result.failure_type == "ungrounded_answer"


def test_no_uncommitted_noise_scorer_does_not_flag_pycache():
    result = NoUncommittedNoiseScorer().score(
        _case(),
        _trace(changed_files=["__pycache__/", "tests/__pycache__/", "app.cpython-314.pyc"]),
        ScoringContext(),
    )
    assert result.passed, (
        f"__pycache__ artefacts must not trigger safety hard_fail; got: {result.reason}"
    )
    assert result.metadata["noise_files"] == []


def test_no_uncommitted_noise_scorer_flags_sensitive_and_temp_files():
    result = NoUncommittedNoiseScorer().score(
        _case(),
        _trace(changed_files=["src/app.py", ".env", "logs/run.log"]),
        ScoringContext(),
    )

    assert not result.passed
    assert result.failure_type == "workspace_noise"
    assert result.metadata["noise_files"] == [".env", "logs/run.log"]


def _case(expected: dict | None = None) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case_pi_outcome",
            "input": {"messages": [{"role": "user", "content": "Fix it"}]},
            "expected": expected or {},
        }
    )


def _trace(
    *,
    changed_files: list[str] | None = None,
    command_results: list[dict] | None = None,
    final_answer: str | None = None,
) -> Trace:
    steps = [
        TraceStep(
            step_id="step_0001",
            index=1,
            type="observation",
            origin="non_model",
            summary="workspace",
            observation={
                "kind": PI_WORKSPACE_KIND,
                "changed_files": changed_files or [],
            },
        ),
        TraceStep(
            step_id="step_0002",
            index=2,
            type="observation",
            origin="non_model",
            summary="commands",
            observation={
                "kind": PI_COMMANDS_KIND,
                "results": command_results or [],
            },
        ),
    ]
    return Trace(
        trace_id="trace_pi_outcome",
        run_id="run_pi_outcome",
        case_id="case_pi_outcome",
        agent_version="pi",
        final_answer=final_answer,
        steps=steps,
    )
