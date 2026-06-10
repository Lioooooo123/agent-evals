"""Deterministic tool trajectory checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_evals.scorers.base import ScoreResult, ScoringContext
from agent_evals.traces.schema import EvalCase, ExpectedToolCall, Trace, TraceStep


@dataclass(frozen=True)
class ActualToolCall:
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str | None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "tool_call_id": self.tool_call_id,
        }


class ToolTrajectoryScorer:
    name = "tool_call_accuracy"

    def score(
        self,
        case: EvalCase,
        trace: Trace,
        context: ScoringContext,
    ) -> ScoreResult:
        expected = case.expected.tool_calls
        actual = _model_tool_calls(trace.steps)

        if not expected:
            return ScoreResult(
                name=self.name,
                score=1.0,
                passed=True,
                reason="no expected tool calls configured",
                metadata={"actual_count": len(actual), "expected_count": 0},
            )

        match_mode = _match_mode(expected)
        matched = _matches(match_mode, expected, actual)
        diagnostic = _diagnose_mismatch(expected, actual, match_mode) if not matched else None
        return ScoreResult(
            name=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            reason=(
                f"actual tool trajectory matches expected ({match_mode})"
                if matched
                else diagnostic["reason"]
            ),
            failure_type="none" if matched else diagnostic["failure_type"],
            metadata={
                "match_mode": match_mode,
                "expected_count": len(expected),
                "actual_count": len(actual),
                "actual_tool_names": [call.tool_name for call in actual],
                "diagnostic": diagnostic,
                "mode_semantics": {
                    "strict": "same length, same order",
                    "unordered": "same length, any order",
                    "subset": "expected is an ordered subsequence of actual",
                    "superset": "actual contains no calls outside expected",
                },
            },
        )


def _model_tool_calls(steps: list[TraceStep]) -> list[ActualToolCall]:
    calls: list[ActualToolCall] = []
    for step in steps:
        if step.type != "tool_call" or step.origin != "model" or step.tool_call is None:
            continue
        calls.append(
            ActualToolCall(
                tool_name=step.tool_call.tool_name,
                arguments=step.tool_call.arguments,
                tool_call_id=step.tool_call.tool_call_id,
            )
        )
    return calls


def _match_mode(expected: list[ExpectedToolCall]) -> str:
    mode = expected[0].match_mode
    return "strict" if mode == "exact" else mode


def _expected_to_metadata(expected: ExpectedToolCall) -> dict[str, Any]:
    return {
        "tool_name": expected.tool_name,
        "arguments": expected.arguments,
        "match_mode": expected.match_mode,
        "argument_match_mode": expected.argument_match_mode,
    }


def _diagnose_mismatch(
    expected: list[ExpectedToolCall],
    actual: list[ActualToolCall],
    match_mode: str,
) -> dict[str, Any]:
    expected_names = [call.tool_name for call in expected]
    actual_names = [call.tool_name for call in actual]

    if sorted(expected_names) != sorted(actual_names):
        missing = _multiset_difference(expected_names, actual_names)
        unexpected = _multiset_difference(actual_names, expected_names)
        return {
            "failure_type": "tool_selection",
            "reason": _selection_reason(missing, unexpected),
            "missing_tools": missing,
            "unexpected_tools": unexpected,
            "expected_calls": [_expected_to_metadata(call) for call in expected],
            "actual_calls": [call.to_metadata() for call in actual],
        }

    if len(expected) == len(actual) and expected_names == actual_names:
        for expected_call, actual_call in zip(expected, actual, strict=True):
            if not _arguments_match(expected_call, actual_call):
                diff = _argument_diff(expected_call.arguments, actual_call.arguments)
                return {
                    "failure_type": "tool_arguments",
                    "reason": _argument_reason(expected_call.tool_name, diff),
                    "expected_call": _expected_to_metadata(expected_call),
                    "actual_call": actual_call.to_metadata(),
                    "argument_diff": diff,
                }

    if _match_unordered(expected, actual):
        return {
            "failure_type": "tool_order",
            "reason": "tool calls have the expected names and arguments but appear in the wrong order",
            "expected_calls": [_expected_to_metadata(call) for call in expected],
            "actual_calls": [call.to_metadata() for call in actual],
        }

    if match_mode in {"unordered", "subset", "superset"}:
        argument_mismatch = _find_unordered_argument_mismatch(expected, actual)
        if argument_mismatch is not None:
            expected_call, actual_call = argument_mismatch
            diff = _argument_diff(expected_call.arguments, actual_call.arguments)
            return {
                "failure_type": "tool_arguments",
                "reason": _argument_reason(expected_call.tool_name, diff),
                "expected_call": _expected_to_metadata(expected_call),
                "actual_call": actual_call.to_metadata(),
                "argument_diff": diff,
            }

    return {
        "failure_type": "tool_order",
        "reason": f"actual tool trajectory does not match expected ({match_mode})",
        "expected_calls": [_expected_to_metadata(call) for call in expected],
        "actual_calls": [call.to_metadata() for call in actual],
    }


def _multiset_difference(left: list[str], right: list[str]) -> list[str]:
    remaining = list(right)
    diff: list[str] = []
    for item in left:
        if item in remaining:
            remaining.remove(item)
        else:
            diff.append(item)
    return diff


def _selection_reason(missing: list[str], unexpected: list[str]) -> str:
    parts: list[str] = []
    if missing:
        parts.append(f"missing expected tool(s): {missing}")
    if unexpected:
        parts.append(f"unexpected tool(s): {unexpected}")
    return "; ".join(parts)


def _matches(
    match_mode: str,
    expected: list[ExpectedToolCall],
    actual: list[ActualToolCall],
) -> bool:
    if match_mode == "strict":
        return len(expected) == len(actual) and all(
            _call_matches(expected_call, actual_call)
            for expected_call, actual_call in zip(expected, actual, strict=True)
        )
    if match_mode == "unordered":
        return len(expected) == len(actual) and _match_unordered(expected, actual)
    if match_mode == "subset":
        return _match_ordered_subset(expected, actual)
    if match_mode == "superset":
        return _match_unordered(
            [ExpectedToolCall(tool_name=call.tool_name, arguments=call.arguments) for call in actual],
            [
                ActualToolCall(
                    tool_name=expected_call.tool_name,
                    arguments=expected_call.arguments,
                    tool_call_id=None,
                )
                for expected_call in expected
            ],
        )
    raise ValueError(f"unsupported match_mode: {match_mode}")


def _match_ordered_subset(
    expected: list[ExpectedToolCall],
    actual: list[ActualToolCall],
) -> bool:
    actual_index = 0
    for expected_call in expected:
        while actual_index < len(actual):
            if _call_matches(expected_call, actual[actual_index]):
                actual_index += 1
                break
            actual_index += 1
        else:
            return False
    return True


def _match_unordered(
    expected: list[ExpectedToolCall],
    actual: list[ActualToolCall],
) -> bool:
    remaining = list(actual)
    for expected_call in expected:
        match_index = next(
            (
                index
                for index, actual_call in enumerate(remaining)
                if _call_matches(expected_call, actual_call)
            ),
            None,
        )
        if match_index is None:
            return False
        remaining.pop(match_index)
    return True


def _call_matches(expected: ExpectedToolCall, actual: ActualToolCall) -> bool:
    if expected.tool_name != actual.tool_name:
        return False
    return _arguments_match(expected, actual)


def _arguments_match(expected: ExpectedToolCall, actual: ActualToolCall) -> bool:
    if expected.argument_match_mode == "subset" or expected.match_mode == "subset":
        return _dict_is_subset(expected.arguments, actual.arguments)
    return expected.arguments == actual.arguments


def _dict_is_subset(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key not in actual:
            return False
        actual_value = actual[key]
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            if not _dict_is_subset(expected_value, actual_value):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _find_unordered_argument_mismatch(
    expected: list[ExpectedToolCall],
    actual: list[ActualToolCall],
) -> tuple[ExpectedToolCall, ActualToolCall] | None:
    actual_by_name: dict[str, list[ActualToolCall]] = {}
    for actual_call in actual:
        actual_by_name.setdefault(actual_call.tool_name, []).append(actual_call)

    for expected_call in expected:
        for actual_call in actual_by_name.get(expected_call.tool_name, []):
            if not _arguments_match(expected_call, actual_call):
                return expected_call, actual_call
    return None


def _argument_diff(expected: dict[str, Any], actual: dict[str, Any]) -> list[dict[str, Any]]:
    keys = sorted(set(expected) | set(actual))
    return [
        {"key": key, "expected": expected.get(key), "actual": actual.get(key)}
        for key in keys
        if expected.get(key) != actual.get(key)
    ]


def _argument_reason(tool_name: str, diff: list[dict[str, Any]]) -> str:
    if not diff:
        return f"tool {tool_name!r} arguments differ"
    first = diff[0]
    return (
        f"tool {tool_name!r} arg mismatch: expected {first['key']}="
        f"{first['expected']}, got {first['actual']}"
    )
