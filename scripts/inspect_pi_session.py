#!/usr/bin/env python3
"""Inspect a Pi session JSONL file without implementing a full parser."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            record["_line"] = line_number
            records.append(record)
    return records


def compact(value: Any, limit: int = 220) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    top_types: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    content_types: Counter[str] = Counter()
    assistant_content_types: Counter[str] = Counter()
    tool_name_counts: Counter[str] = Counter()
    tool_result_keys: Counter[str] = Counter()
    usage_keys: Counter[str] = Counter()
    cost_keys: Counter[str] = Counter()
    parent_children: dict[str | None, list[str]] = defaultdict(list)
    records_by_id: dict[str, dict[str, Any]] = {}
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    bash_executions: list[dict[str, Any]] = []

    for record in records:
        record_type = record.get("type")
        top_types[str(record_type)] += 1
        record_id = record.get("id")
        if isinstance(record_id, str):
            records_by_id[record_id] = record
            parent_children[record.get("parentId")].append(record_id)

        message = record.get("message")
        if not isinstance(message, dict):
            continue

        role = message.get("role")
        roles[str(role)] += 1
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = str(block.get("type"))
                content_types[block_type] += 1
                if role == "assistant":
                    assistant_content_types[block_type] += 1
                if role == "assistant" and block.get("type") == "toolCall":
                    tool_name_counts[str(block.get("name"))] += 1
                    tool_calls.append(
                        {
                            "line": record["_line"],
                            "messageId": record.get("id"),
                            "parentId": record.get("parentId"),
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "arguments": block.get("arguments"),
                            "partialJson": block.get("partialJson"),
                        }
                    )

        if role == "toolResult":
            tool_result_keys.update(message.keys())
            tool_results.append(
                {
                    "line": record["_line"],
                    "messageId": record.get("id"),
                    "parentId": record.get("parentId"),
                    "toolCallId": message.get("toolCallId"),
                    "toolName": message.get("toolName"),
                    "content": message.get("content"),
                    "isError": message.get("isError"),
                    "details": message.get("details"),
                    "keys": sorted(message.keys()),
                }
            )

        if role == "bashExecution":
            bash_executions.append(
                {
                    "line": record["_line"],
                    "messageId": record.get("id"),
                    "parentId": record.get("parentId"),
                    "keys": sorted(message.keys()),
                    "command": message.get("command"),
                    "output": message.get("output"),
                    "exitCode": message.get("exitCode"),
                    "toolCallId": message.get("toolCallId"),
                    "timestamp": message.get("timestamp"),
                }
            )

        if role == "assistant" and isinstance(message.get("usage"), dict):
            usage = message["usage"]
            usage_keys.update(usage.keys())
            cost = usage.get("cost")
            if isinstance(cost, dict):
                cost_keys.update(cost.keys())

    tool_calls_by_id = {
        call["id"]: call for call in tool_calls if isinstance(call.get("id"), str)
    }
    result_pairs = [
        {
            "toolCallId": result.get("toolCallId"),
            "toolName": result.get("toolName"),
            "resultLine": result.get("line"),
            "callLine": tool_calls_by_id.get(result.get("toolCallId"), {}).get("line"),
            "matched": result.get("toolCallId") in tool_calls_by_id,
        }
        for result in tool_results
    ]

    bash_tool_calls = [call for call in tool_calls if call.get("name") == "bash"]
    bash_tool_results = [
        result for result in tool_results if result.get("toolName") == "bash"
    ]
    bash_result_ids = {
        result.get("toolCallId")
        for result in bash_tool_results
        if isinstance(result.get("toolCallId"), str)
    }
    bash_call_ids = {
        call.get("id") for call in bash_tool_calls if isinstance(call.get("id"), str)
    }
    bash_execution_tool_ids = {
        item.get("toolCallId")
        for item in bash_executions
        if isinstance(item.get("toolCallId"), str)
    }

    branch_points = {
        parent: children
        for parent, children in parent_children.items()
        if parent is not None and len(children) > 1
    }

    return {
        "line_count": len(records),
        "top_types": top_types,
        "roles": roles,
        "content_types": content_types,
        "assistant_content_types": assistant_content_types,
        "tool_name_counts": tool_name_counts,
        "tool_result_keys": tool_result_keys,
        "usage_keys": usage_keys,
        "cost_keys": cost_keys,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "bash_executions": bash_executions,
        "result_pairs": result_pairs,
        "bash_tool_calls": bash_tool_calls,
        "bash_tool_results": bash_tool_results,
        "bash_call_ids": bash_call_ids,
        "bash_result_ids": bash_result_ids,
        "bash_execution_tool_ids": bash_execution_tool_ids,
        "branch_points": branch_points,
        "root_children": parent_children.get(None, []),
        "max_children": max((len(children) for children in parent_children.values()), default=0),
    }


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    for key, count in counter.most_common():
        print(f"  {key}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="cases/fixtures/pi_session_sample.jsonl",
        help="Path to a Pi session JSONL file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a compact JSON summary instead of human-readable output.",
    )
    args = parser.parse_args()

    path = Path(args.path)
    records = load_jsonl(path)
    summary = summarize(records)

    if args.json:
        serializable = {
            key: (dict(value) if isinstance(value, Counter) else value)
            for key, value in summary.items()
            if key
            not in {
                "bash_call_ids",
                "bash_result_ids",
                "bash_execution_tool_ids",
                "branch_points",
                "root_children",
            }
        }
        serializable["bash_call_ids"] = sorted(summary["bash_call_ids"])
        serializable["bash_result_ids"] = sorted(summary["bash_result_ids"])
        serializable["bash_execution_tool_ids"] = sorted(
            summary["bash_execution_tool_ids"]
        )
        serializable["branch_points"] = {
            str(key): value for key, value in summary["branch_points"].items()
        }
        serializable["root_children"] = summary["root_children"]
        print(json.dumps(serializable, ensure_ascii=False, indent=2))
        return

    print(f"File: {path}")
    print(f"Lines: {summary['line_count']}")
    print_counter("Top-level type counts:", summary["top_types"])
    print_counter("Message role counts:", summary["roles"])
    print_counter("Content block type counts:", summary["content_types"])
    print_counter(
        "Assistant content block type counts:", summary["assistant_content_types"]
    )
    print_counter("Assistant tool name counts:", summary["tool_name_counts"])
    print_counter("ToolResult key counts:", summary["tool_result_keys"])
    print_counter("Assistant usage keys:", summary["usage_keys"])
    print_counter("Usage cost keys:", summary["cost_keys"])
    print()
    print(f"Assistant tool calls: {len(summary['tool_calls'])}")
    print(f"Tool results: {len(summary['tool_results'])}")
    print(f"Bash tool calls: {len(summary['bash_tool_calls'])}")
    print(f"Bash tool results: {len(summary['bash_tool_results'])}")
    print(f"Bash execution messages: {len(summary['bash_executions'])}")
    print(f"ToolResult matched to toolCall by id: {sum(1 for p in summary['result_pairs'] if p['matched'])}/{len(summary['result_pairs'])}")
    print(f"Branch points: {len(summary['branch_points'])}")
    print(f"Root children: {len(summary['root_children'])}")
    print(f"Max children for one parent: {summary['max_children']}")
    print()

    for index, call in enumerate(summary["tool_calls"][:2], 1):
        print(f"Tool call example {index}:")
        print(f"  line: {call['line']}")
        print(f"  id: {call['id']}")
        print(f"  name: {call['name']}")
        print(f"  arguments: {compact(call['arguments'])}")
    if summary["tool_results"]:
        result = summary["tool_results"][0]
        print("Tool result example 1:")
        print(f"  line: {result['line']}")
        print(f"  toolCallId: {result['toolCallId']}")
        print(f"  toolName: {result['toolName']}")
        print(f"  keys: {', '.join(result['keys'])}")
        print(f"  isError: {result['isError']}")
        print(f"  content: {compact(result['content'])}")
        print(f"  details: {compact(result['details'])}")
    if summary["bash_executions"]:
        bash = summary["bash_executions"][0]
        print("Bash execution example 1:")
        print(f"  line: {bash['line']}")
        print(f"  keys: {', '.join(bash['keys'])}")
        print(f"  toolCallId: {bash['toolCallId']}")
        print(f"  command: {compact(bash['command'])}")
        print(f"  exitCode: {bash['exitCode']}")
        print(f"  output: {compact(bash['output'])}")


if __name__ == "__main__":
    main()
