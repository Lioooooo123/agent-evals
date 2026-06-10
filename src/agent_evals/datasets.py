"""Dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from agent_evals.traces.schema import EvalCase


class DatasetLoadError(ValueError):
    """Raised when an EvalCase JSONL file cannot be loaded."""


class DuplicateEvalCaseIdError(DatasetLoadError):
    """Raised when an EvalCase JSONL file contains duplicate ids."""


def load_eval_cases_jsonl(path: str | Path) -> list[EvalCase]:
    """Load and validate EvalCase rows from a JSONL file."""

    jsonl_path = Path(path)
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetLoadError(
                    f"{jsonl_path}:{line_number}: invalid JSON: {exc}"
                ) from exc

            try:
                case = EvalCase.model_validate(payload)
            except ValidationError as exc:
                raise DatasetLoadError(
                    f"{jsonl_path}:{line_number}: invalid EvalCase: {exc}"
                ) from exc

            if case.id in seen_ids:
                raise DuplicateEvalCaseIdError(
                    f"{jsonl_path}:{line_number}: duplicate EvalCase id {case.id!r}"
                )
            seen_ids.add(case.id)
            cases.append(case)

    return cases


def dump_eval_cases_jsonl(cases: Iterable[EvalCase], path: str | Path) -> None:
    """Write EvalCase rows as JSONL. Mostly useful for tests and fixtures."""

    jsonl_path = Path(path)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json(by_alias=True))
            handle.write("\n")
