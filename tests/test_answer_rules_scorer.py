from __future__ import annotations

from agent_evals.scorers import AnswerRuleScorer, ScoringContext
from agent_evals.traces.schema import EvalCase, Trace


def _case(expected: dict) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case_answer",
            "input": {"messages": [{"role": "user", "content": "answer"}]},
            "expected": expected,
        }
    )


def _trace(final_answer: str) -> Trace:
    return Trace(
        trace_id="trace_answer",
        run_id="run_answer",
        case_id="case_answer",
        agent_version="test",
        final_answer=final_answer,
    )


def test_answer_contains_hit_and_miss():
    scorer = AnswerRuleScorer()

    hit = scorer.score(
        _case({"answer_contains": ["order", "delivered"]}),
        _trace("The order was delivered today."),
        ScoringContext(),
    )
    miss = scorer.score(
        _case({"answer_contains": ["refund"]}),
        _trace("The order was delivered today."),
        ScoringContext(),
    )

    assert hit.passed
    assert hit.score == 1.0
    assert not miss.passed
    assert miss.score == 0.0


def test_answer_contains_is_advisory_when_commands_present():
    scorer = AnswerRuleScorer()

    case_with_commands = EvalCase.model_validate(
        {
            "id": "case_advisory",
            "input": {"messages": [{"role": "user", "content": "fix it"}]},
            "expected": {
                "answer_contains": ["fixed"],
                "commands": [{"cmd": "pytest", "cwd": ".", "timeout_s": 30, "must_pass": True}],
            },
        }
    )
    result = scorer.score(
        case_with_commands,
        _trace("All tests now pass."),  # doesn't contain "fixed"
        ScoringContext(),
    )
    assert result.passed, (
        "answer_contains should be advisory (not a hard gate) when expected.commands is present"
    )
    assert result.metadata["missing_contains"] == ["fixed"]


def test_answer_must_not_contain_hit_and_miss():
    scorer = AnswerRuleScorer()

    passed = scorer.score(
        _case({"answer_must_not_contain": ["tracking number"]}),
        _trace("The order was delivered today."),
        ScoringContext(),
    )
    failed = scorer.score(
        _case({"answer_must_not_contain": ["tracking number"]}),
        _trace("The tracking number is fabricated."),
        ScoringContext(),
    )

    assert passed.passed
    assert not failed.passed
    assert "forbidden" in failed.reason


def test_answer_regex_hit_and_miss():
    scorer = AnswerRuleScorer()

    hit = scorer.score(
        _case({"answer_regex": [r"order\s+\w+"]}),
        _trace("order A123 is in transit"),
        ScoringContext(),
    )
    miss = scorer.score(
        _case({"answer_regex": [r"order\s+\d{6}"]}),
        _trace("order A123 is in transit"),
        ScoringContext(),
    )

    assert hit.passed
    assert not miss.passed
    assert "regex" in miss.reason


def test_answer_json_schema_pass_and_fail():
    scorer = AnswerRuleScorer()
    schema = {
        "type": "object",
        "required": ["status"],
        "properties": {"status": {"type": "string"}},
    }

    passed = scorer.score(
        _case({"answer_json_schema": schema}),
        _trace('{"status": "ok"}'),
        ScoringContext(),
    )
    failed = scorer.score(
        _case({"answer_json_schema": schema}),
        _trace('{"state": "ok"}'),
        ScoringContext(),
    )

    assert passed.passed
    assert not failed.passed
    assert "schema" in failed.reason
