"""Pi session JSONL to Trace conversion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_evals.traces.schema import Trace, TraceMetrics, TraceStep, TraceToolCall


JsonDict = dict[str, Any]


class PiSessionParseError(ValueError):
    """Raised when a Pi session JSONL file cannot be parsed."""


@dataclass(frozen=True)
class SessionRecord:
    line_number: int
    payload: JsonDict

    @property
    def id(self) -> str | None:
        value = self.payload.get("id")
        return value if isinstance(value, str) else None

    @property
    def parent_id(self) -> str | None:
        value = self.payload.get("parentId")
        return value if isinstance(value, str) else None


def parse_pi_session_jsonl(
    path: str | Path,
    *,
    run_id: str | None = None,
    case_id: str | None = None,
) -> Trace:
    """Parse a Pi session JSONL file into a Trace.

    This is intentionally a focused M1 parser for the observed Pi session shape.
    It maps model tool calls through the assistant toolCall/toolResult channel and
    treats standalone bashExecution messages as non-model audit steps.
    """

    jsonl_path = Path(path)
    records = _load_records(jsonl_path)
    if not records:
        raise PiSessionParseError(f"{jsonl_path}: empty session")

    session_record = next(
        (record.payload for record in records if record.payload.get("type") == "session"),
        {},
    )
    session_id = str(session_record.get("id") or jsonl_path.stem)
    main_chain = _select_main_chain(records)

    trace = Trace(
        trace_id=f"trace_{session_id}",
        run_id=run_id or f"run_{session_id}",
        case_id=case_id or session_id,
        agent_version=_agent_version(main_chain),
        started_at=_parse_timestamp(session_record.get("timestamp")),
        ended_at=_parse_timestamp(main_chain[-1].payload.get("timestamp")) if main_chain else None,
    )

    final_answer: str | None = None
    step_index = 0
    tool_calls_by_id: dict[str, TraceStep] = {}

    for record in main_chain:
        payload = record.payload
        message = payload.get("message")
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        timestamp = _parse_timestamp(payload.get("timestamp"))

        if role == "assistant":
            _add_usage(trace.metrics, message.get("usage"))
            emitted_for_message = False
            for block in _content_blocks(message):
                block_type = block.get("type")
                if block_type == "text":
                    text = str(block.get("text", ""))
                    if not text:
                        continue
                    step_index += 1
                    trace.steps.append(
                        TraceStep(
                            step_id=f"step_{step_index:04d}",
                            index=step_index,
                            type="llm",
                            timestamp=timestamp,
                            summary=text,
                        )
                    )
                    final_answer = text
                    emitted_for_message = True
                elif block_type == "toolCall":
                    tool_call_id = _as_str(block.get("id"))
                    tool_name = _as_str(block.get("name")) or "unknown"
                    arguments = block.get("arguments")
                    step_index += 1
                    step = TraceStep(
                        step_id=f"step_{step_index:04d}",
                        index=step_index,
                        type="tool_call",
                        timestamp=timestamp,
                        summary=f"{tool_name} tool call",
                        origin="model",
                        tool_call=TraceToolCall(
                            tool_name=tool_name,
                            arguments=arguments if isinstance(arguments, dict) else {},
                            tool_call_id=tool_call_id,
                        ),
                    )
                    trace.steps.append(step)
                    if tool_call_id:
                        tool_calls_by_id[tool_call_id] = step
                    emitted_for_message = True

            if not emitted_for_message:
                continue

        elif role == "toolResult":
            tool_call_id = _as_str(message.get("toolCallId"))
            tool_name = _as_str(message.get("toolName")) or "unknown"
            step_index += 1
            trace.steps.append(
                TraceStep(
                    step_id=f"step_{step_index:04d}",
                    index=step_index,
                    type="observation",
                    timestamp=timestamp,
                    summary=f"{tool_name} observation",
                    origin="model",
                    observation={
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "content": message.get("content"),
                        "is_error": bool(message.get("isError", False)),
                        "details": message.get("details"),
                    },
                )
            )

        elif role == "bashExecution":
            command = str(message.get("command", ""))
            step_index += 1
            trace.steps.append(
                TraceStep(
                    step_id=f"step_{step_index:04d}",
                    index=step_index,
                    type="tool_call",
                    timestamp=timestamp,
                    summary="non-model bash execution",
                    origin="non_model",
                    tool_call=TraceToolCall(
                        tool_name="bash",
                        arguments={"command": command},
                    ),
                    observation={
                        "output": message.get("output"),
                        "exit_code": message.get("exitCode"),
                        "cancelled": message.get("cancelled"),
                        "truncated": message.get("truncated"),
                        "exclude_from_context": message.get("excludeFromContext"),
                    },
                )
            )

    trace.final_answer = final_answer
    return trace


def _load_records(path: Path) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise PiSessionParseError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise PiSessionParseError(
                    f"{path}:{line_number}: expected object per JSONL row"
                )
            records.append(SessionRecord(line_number=line_number, payload=payload))
    return records


def _select_main_chain(records: list[SessionRecord]) -> list[SessionRecord]:
    records_by_id = {record.id: record for record in records if record.id}
    parent_ids = {record.parent_id for record in records if record.parent_id}
    leaves = [
        record
        for record in records
        if record.id and record.id not in parent_ids and record.payload.get("type") != "session"
    ]
    if not leaves:
        return records

    leaf = max(leaves, key=lambda record: record.line_number)
    chain: list[SessionRecord] = []
    current: SessionRecord | None = leaf
    seen: set[str] = set()

    while current is not None:
        current_id = current.id
        if current_id is None or current_id in seen:
            break
        seen.add(current_id)
        chain.append(current)
        parent_id = current.parent_id
        current = records_by_id.get(parent_id) if parent_id else None

    return list(reversed(chain))


def _agent_version(records: list[SessionRecord]) -> str:
    for record in records:
        payload = record.payload
        if payload.get("type") == "model_change":
            provider = payload.get("provider")
            model_id = payload.get("modelId")
            if provider and model_id:
                return f"{provider}/{model_id}"
    for record in records:
        message = record.payload.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            provider = message.get("provider")
            model = message.get("model")
            if provider and model:
                return f"{provider}/{model}"
    return "unknown"


def _content_blocks(message: JsonDict) -> list[JsonDict]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _add_usage(metrics: TraceMetrics, usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    metrics.input_tokens += int(usage.get("input") or 0)
    metrics.output_tokens += int(usage.get("output") or 0)
    metrics.cache_read_tokens += int(usage.get("cacheRead") or 0)
    metrics.cache_write_tokens += int(usage.get("cacheWrite") or 0)
    metrics.total_tokens += int(usage.get("totalTokens") or 0)
    cost = usage.get("cost")
    if isinstance(cost, dict):
        metrics.cost_usd += float(cost.get("total") or 0.0)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None
