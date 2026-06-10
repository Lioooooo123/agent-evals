from __future__ import annotations

import json

from agent_evals.scorers import ScoringContext, TaskSuccessJudgeScorer, judge_dimension_results
from agent_evals.scorers.judge import JudgeRubricConfig
from agent_evals.traces.schema import EvalCase, Trace


class FakeJudgeClient:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt: str, config: JudgeRubricConfig) -> str:
        self.calls += 1
        return self.responses.pop(0)


def _config() -> JudgeRubricConfig:
    return JudgeRubricConfig.model_validate(
        {
            "version": "test-rubric-v1",
            "provider": {
                "name": "anthropic",
                "model": "test-model",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
            "rubric": {"overall_score": "score the task"},
        }
    )


def _case() -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "case_judge",
            "input": {"messages": [{"role": "user", "content": "Do the task"}]},
            "expected": {"answer_contains": ["done"]},
        }
    )


def _trace() -> Trace:
    return Trace(
        trace_id="trace_judge",
        run_id="run_judge",
        case_id="case_judge",
        agent_version="test",
        final_answer="done",
    )


def _valid_response(**overrides) -> str:
    payload = {
        "goal_completion": 1.0,
        "tool_use": 1.0,
        "grounding": 0.8,
        "efficiency": 0.7,
        "safety": 1.0,
        "overall_score": 0.9,
        "pass": True,
        "failure_type": "none",
        "reason": "task completed",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_task_success_judge_valid_json_scores_normally():
    client = FakeJudgeClient([_valid_response()])
    scorer = TaskSuccessJudgeScorer(_config(), client=client)

    result = scorer.score(_case(), _trace(), ScoringContext())
    dimensions = judge_dimension_results(result)

    assert result.name == "task_success"
    assert result.passed
    assert result.score == 0.9
    assert result.metadata["rubric_version"] == "test-rubric-v1"
    assert {score.name for score in dimensions} == {"grounding", "safety"}
    assert client.calls == 1


def test_task_success_judge_invalid_json_retries_once():
    client = FakeJudgeClient(["not json", _valid_response(overall_score=0.75)])
    scorer = TaskSuccessJudgeScorer(_config(), client=client)

    result = scorer.score(_case(), _trace(), ScoringContext())

    assert result.passed
    assert result.score == 0.75
    assert result.metadata["attempts"] == 2
    assert client.calls == 2


def test_task_success_judge_two_invalid_outputs_return_judge_error():
    client = FakeJudgeClient(["not json", "still not json"])
    scorer = TaskSuccessJudgeScorer(_config(), client=client)

    result = scorer.score(_case(), _trace(), ScoringContext())

    assert not result.passed
    assert result.score == 0.0
    assert result.failure_type == "judge_error"
    assert client.calls == 2
