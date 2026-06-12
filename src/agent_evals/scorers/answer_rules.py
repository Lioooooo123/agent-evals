"""Rule-based final answer checks."""

from __future__ import annotations

import json
import re

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema

from agent_evals.scorers.base import ScoreResult, ScoringContext
from agent_evals.traces.schema import EvalCase, Trace


class AnswerRuleScorer:
    name = "final_answer_correctness"

    def score(
        self,
        case: EvalCase,
        trace: Trace,
        context: ScoringContext,
    ) -> ScoreResult:
        answer = trace.final_answer or ""
        failures: list[str] = []

        missing_contains = [
            expected
            for expected in case.expected.answer_contains
            if expected not in answer
        ]
        # When the case has expected commands, execution results are the authoritative
        # success signal; answer_contains becomes advisory only (not a hard gate).
        has_commands = bool(case.expected.commands)
        if missing_contains and not has_commands:
            failures.append(f"missing required text: {missing_contains}")

        forbidden_hits = [
            forbidden
            for forbidden in case.expected.answer_must_not_contain
            if forbidden in answer
        ]
        if forbidden_hits:
            failures.append(f"contains forbidden text: {forbidden_hits}")

        regex_failures = [
            pattern
            for pattern in case.expected.answer_regex
            if re.search(pattern, answer) is None
        ]
        if regex_failures:
            failures.append(f"regex did not match: {regex_failures}")

        schema_error: str | None = None
        if case.expected.answer_json_schema is not None:
            try:
                payload = json.loads(answer)
                validate_json_schema(payload, case.expected.answer_json_schema)
            except json.JSONDecodeError as exc:
                schema_error = f"answer is not valid JSON: {exc.msg}"
            except JsonSchemaValidationError as exc:
                schema_error = f"answer JSON schema validation failed: {exc.message}"
            if schema_error:
                failures.append(schema_error)

        passed = not failures
        return ScoreResult(
            name=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="answer rules passed" if passed else "; ".join(failures),
            failure_type="none" if passed else "format_error",
            metadata={
                "missing_contains": missing_contains,
                "forbidden_hits": forbidden_hits,
                "regex_failures": regex_failures,
                "schema_error": schema_error,
            },
        )
