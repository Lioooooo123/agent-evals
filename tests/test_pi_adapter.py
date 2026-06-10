from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agent_evals.adapters.pi import (
    PI_COMMANDS_KIND,
    PI_SESSION_KIND,
    PI_WORKSPACE_KIND,
    PiAgentAdapter,
    _git_state,
    _run_expected_commands,
    discover_pi_session,
    prepare_workspace,
)
from agent_evals.scorers import CommandPassScorer, ScoringContext, WorkspaceDiffScorer
from agent_evals.traces.schema import EvalCase, ExpectedCommand


def test_prepare_workspace_copies_source_and_initializes_git_baseline(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "app.py").write_text("value = 1\n", encoding="utf-8")
    case = _case(source)

    workspace = prepare_workspace(case, run_id="run_1", base_work_dir=tmp_path / "runs")

    assert workspace != source
    assert (workspace / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert (workspace / ".git").exists()
    assert not (source / ".git").exists()


def test_prepare_workspace_ignores_source_git_metadata_and_dirty_state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init"], cwd=source, capture_output=True, text=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "agent-evals-test"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "agent-evals-test@example.invalid"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    (source / "tracked.py").write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=source, capture_output=True, text=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "clean baseline"],
        cwd=source,
        capture_output=True,
        text=True,
        check=True,
    )
    (source / "tracked.py").write_text("dirty\n", encoding="utf-8")

    workspace = prepare_workspace(_case(source), run_id="run_dirty", base_work_dir=tmp_path / "runs")

    assert (workspace / ".git").exists()
    assert (workspace / "tracked.py").read_text(encoding="utf-8") == "dirty\n"
    assert _git_state(workspace)["changed_files"] == []


def test_expected_command_cwd_cannot_escape_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker.txt").write_text("secret\n", encoding="utf-8")

    results = _run_expected_commands(
        [
            ExpectedCommand(
                cmd="pwd && test -f marker.txt",
                cwd=str(outside),
                timeout_s=5,
                must_pass=True,
            )
        ],
        workspace,
    )

    assert results[0]["exit_code"] != 0
    assert str(outside) not in results[0]["stdout"]
    assert "escapes workspace" in results[0]["stderr"]


def test_discover_pi_session_supports_explicit_and_session_dir_latest(tmp_path):
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    older = session_dir / "older.jsonl"
    newer = session_dir / "newer.jsonl"
    older.write_text('{"type":"session","id":"old"}\n', encoding="utf-8")
    newer.write_text('{"type":"session","id":"new"}\n', encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    latest_case = _case(tmp_path, pi={"session_discovery": "session_dir_latest"})
    assert discover_pi_session(latest_case, session_dir) == (newer, None)

    explicit_case = _case(
        tmp_path,
        pi={"session_discovery": "explicit", "session_jsonl": str(older)},
    )
    assert discover_pi_session(explicit_case, session_dir) == (older, None)


def test_pi_adapter_falls_back_to_black_box_when_session_parse_fails(tmp_path):
    source = tmp_path / "target"
    source.mkdir()
    (source / "app.py").write_text("broken\n", encoding="utf-8")
    session_dir = tmp_path / "sessions"
    session_dir.mkdir()
    (session_dir / "bad.jsonl").write_text("not json\n", encoding="utf-8")
    case = _case(
        source,
        pi={"session_dir": str(session_dir), "session_discovery": "session_dir_latest"},
        expected={
            "workspace": {
                "files_changed": ["app.py"],
                "files_forbidden": [".env"],
                "allow_extra_changes": False,
            },
            "commands": [
                {
                    "cmd": f"{sys.executable} -c \"from pathlib import Path; assert Path('app.py').read_text() == 'fixed\\n'\"",
                    "cwd": ".",
                    "timeout_s": 30,
                    "must_pass": True,
                }
            ],
        },
    )

    def fake_runner(*args, **kwargs):
        cwd = Path(kwargs["cwd"])
        (cwd / "app.py").write_text("fixed\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="fixed and tested\n", stderr="")

    adapter = PiAgentAdapter(base_work_dir=tmp_path / "runs", runner=fake_runner)
    output = adapter.run(case, "run_fallback")

    trace = output.trace
    assert trace.final_answer == "fixed and tested"
    assert _observation(trace, PI_SESSION_KIND)["error"].startswith("failed to parse")
    assert _observation(trace, PI_WORKSPACE_KIND)["changed_files"] == ["app.py"]
    assert _observation(trace, PI_COMMANDS_KIND)["results"][0]["exit_code"] == 0
    assert WorkspaceDiffScorer().score(case, trace, ScoringContext()).passed
    assert CommandPassScorer().score(case, trace, ScoringContext()).passed


def _case(source: Path, *, pi: dict | None = None, expected: dict | None = None) -> EvalCase:
    return EvalCase.model_validate(
        {
            "id": "pi_case",
            "input": {"messages": [{"role": "user", "content": "Fix the file."}]},
            "expected": expected or {},
            "metadata": {
                "tags": ["pi"],
                "pi": {
                    "workspace": str(source),
                    "timeout_s": 60,
                    **(pi or {}),
                },
            },
        }
    )


def _observation(trace, kind: str) -> dict:
    return next(
        step.observation
        for step in trace.steps
        if isinstance(step.observation, dict) and step.observation.get("kind") == kind
    )
