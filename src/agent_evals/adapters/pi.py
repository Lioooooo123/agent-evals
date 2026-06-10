"""Pi CLI adapter for black-box code-agent evals."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from agent_evals.adapters.base import AgentOutput
from agent_evals.parsers.pi_session import PiSessionParseError, parse_pi_session_jsonl
from agent_evals.traces.schema import (
    EvalCase,
    ExpectedCommand,
    Trace,
    TraceMetrics,
    TraceStep,
)


PI_PROCESS_KIND = "pi_process"
PI_SESSION_KIND = "pi_session"
PI_WORKSPACE_KIND = "pi_workspace"
PI_COMMANDS_KIND = "pi_expected_commands"

CompletedProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class PiAgentAdapter:
    """Run Pi in print mode inside an isolated workspace."""

    name = "pi"

    def __init__(
        self,
        *,
        pi_binary: str = "pi",
        base_work_dir: str | Path | None = None,
        runner: CompletedProcessRunner = subprocess.run,
    ) -> None:
        self.pi_binary = pi_binary
        self.base_work_dir = Path(base_work_dir) if base_work_dir else None
        self.runner = runner

    def run(self, case: EvalCase, run_id: str) -> AgentOutput:
        started = time.monotonic()
        pi_config = _pi_config(case)
        workspace = prepare_workspace(
            case,
            run_id=run_id,
            base_work_dir=self.base_work_dir,
        )
        session_dir = _session_dir(case, run_id, workspace, pi_config)
        session_dir.mkdir(parents=True, exist_ok=True)

        before_state = _git_state(workspace)
        task = _task_text(case)
        timeout_s = int(pi_config.get("timeout_s", 600))
        stdout = ""
        stderr = ""
        exit_code = 127
        status = "failed"

        try:
            completed = self.runner(
                [self.pi_binary, "-p", task],
                cwd=str(workspace),
                env=_pi_env(session_dir),
                timeout=timeout_s,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = int(completed.returncode)
            status = "completed" if exit_code == 0 else "failed"
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            exit_code = 124
            status = "timeout"
        except FileNotFoundError as exc:
            stderr = str(exc)
            exit_code = 127

        latency_ms = int((time.monotonic() - started) * 1000)
        session_jsonl, session_error = discover_pi_session(case, session_dir)
        trace = _trace_from_session_or_black_box(
            session_jsonl=session_jsonl,
            session_error=session_error,
            case=case,
            run_id=run_id,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            status=status,
        )
        trace.metrics.latency_ms = latency_ms
        if not trace.final_answer and stdout.strip():
            trace.final_answer = stdout.strip()

        command_results = _run_expected_commands(case.expected.commands, workspace)
        after_state = _git_state(workspace)
        _append_adapter_observations(
            trace,
            workspace=workspace,
            source_workspace=_source_workspace(pi_config),
            session_dir=session_dir,
            session_jsonl=session_jsonl,
            session_error=session_error,
            before_state=before_state,
            after_state=after_state,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            command_results=command_results,
        )
        return AgentOutput(trace=trace)


def prepare_workspace(
    case: EvalCase,
    *,
    run_id: str,
    base_work_dir: str | Path | None = None,
) -> Path:
    pi_config = _pi_config(case)
    root = Path(base_work_dir) if base_work_dir else Path(tempfile.mkdtemp(prefix="agent-evals-pi-"))
    workspace = root / run_id / case.id / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    source = _source_workspace(pi_config)
    if source is not None:
        if not source.exists():
            raise FileNotFoundError(f"Pi workspace does not exist: {source}")
        shutil.copytree(source, workspace, ignore=_copy_ignore)
    else:
        workspace.mkdir(parents=True, exist_ok=True)
    _ensure_git_baseline(workspace)
    return workspace


def discover_pi_session(case: EvalCase, session_dir: Path) -> tuple[Path | None, str | None]:
    pi_config = _pi_config(case)
    discovery = str(pi_config.get("session_discovery", "session_dir_latest"))
    if discovery == "none":
        return None, None
    if discovery == "explicit":
        explicit = pi_config.get("session_jsonl")
        if not explicit:
            return None, "metadata.pi.session_jsonl is required for explicit discovery"
        path = Path(str(explicit)).expanduser()
        return (path, None) if path.exists() else (None, f"session JSONL not found: {path}")
    if discovery == "session_dir_latest":
        candidates = [path for path in session_dir.rglob("*.jsonl") if path.is_file()]
        if not candidates:
            return None, f"no session JSONL found under {session_dir}"
        return max(candidates, key=lambda path: path.stat().st_mtime), None
    return None, f"unsupported session_discovery: {discovery}"


def _trace_from_session_or_black_box(
    *,
    session_jsonl: Path | None,
    session_error: str | None,
    case: EvalCase,
    run_id: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    status: str,
) -> Trace:
    if session_jsonl is not None:
        try:
            trace = parse_pi_session_jsonl(session_jsonl, run_id=run_id, case_id=case.id)
            trace.status = status  # type: ignore[assignment]
            return trace
        except PiSessionParseError as exc:
            session_error = f"failed to parse {session_jsonl}: {exc}"

    final_answer = stdout.strip() or None
    trace = Trace(
        trace_id=f"trace_{run_id}_{case.id}",
        run_id=run_id,
        case_id=case.id,
        agent_version="pi/black-box",
        status=status,  # type: ignore[arg-type]
        final_answer=final_answer,
    )
    if final_answer:
        trace.steps.append(
            TraceStep(
                step_id="step_0001",
                index=1,
                type="llm",
                summary=final_answer,
            )
        )
    if session_error:
        _append_observation(
            trace,
            PI_SESSION_KIND,
            "Pi session discovery fallback",
            {"error": session_error, "session_jsonl": str(session_jsonl) if session_jsonl else None},
        )
    if stderr and not final_answer and exit_code != 0:
        trace.final_answer = stderr.strip()
    return trace


def _append_adapter_observations(
    trace: Trace,
    *,
    workspace: Path,
    source_workspace: Path | None,
    session_dir: Path,
    session_jsonl: Path | None,
    session_error: str | None,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    stdout: str,
    stderr: str,
    exit_code: int,
    command_results: list[dict[str, Any]],
) -> None:
    _append_observation(
        trace,
        PI_PROCESS_KIND,
        "Pi process outcome",
        {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "workspace": str(workspace),
            "source_workspace": str(source_workspace) if source_workspace else None,
            "session_dir": str(session_dir),
            "session_jsonl": str(session_jsonl) if session_jsonl else None,
            "session_error": session_error,
        },
    )
    _append_observation(
        trace,
        PI_WORKSPACE_KIND,
        "Pi workspace diff",
        {
            "workspace": str(workspace),
            "before": before_state,
            "after": after_state,
            "changed_files": after_state.get("changed_files", []),
            "diff": after_state.get("diff", ""),
            "diff_stat": after_state.get("diff_stat", ""),
        },
    )
    _append_observation(
        trace,
        PI_COMMANDS_KIND,
        "Pi expected command results",
        {"results": command_results},
    )


def _append_observation(trace: Trace, kind: str, summary: str, payload: dict[str, Any]) -> None:
    index = len(trace.steps) + 1
    trace.steps.append(
        TraceStep(
            step_id=f"step_{index:04d}",
            index=index,
            type="observation",
            summary=summary,
            origin="non_model",
            observation={"kind": kind, **payload},
        )
    )


def _run_expected_commands(commands: Sequence[ExpectedCommand], workspace: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for command in commands:
        started = time.monotonic()
        cwd, cwd_error = _resolve_command_cwd(workspace, command.cwd)
        if cwd_error is not None:
            results.append(
                {
                    "cmd": command.cmd,
                    "cwd": command.cwd,
                    "timeout_s": command.timeout_s,
                    "must_pass": command.must_pass,
                    "exit_code": 126,
                    "stdout": "",
                    "stderr": cwd_error,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "timed_out": False,
                }
            )
            continue
        try:
            completed = subprocess.run(
                command.cmd,
                cwd=str(cwd),
                shell=True,
                timeout=command.timeout_s,
                capture_output=True,
                text=True,
                check=False,
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            results.append(
                {
                    "cmd": command.cmd,
                    "cwd": command.cwd,
                    "timeout_s": command.timeout_s,
                    "must_pass": command.must_pass,
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "latency_ms": latency_ms,
                    "timed_out": False,
                }
            )
        except subprocess.TimeoutExpired as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            results.append(
                {
                    "cmd": command.cmd,
                    "cwd": command.cwd,
                    "timeout_s": command.timeout_s,
                    "must_pass": command.must_pass,
                    "exit_code": 124,
                    "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
                    "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
                    "latency_ms": latency_ms,
                    "timed_out": True,
                }
            )
    return results


def _resolve_command_cwd(workspace: Path, cwd: str) -> tuple[Path | None, str | None]:
    workspace_root = workspace.resolve()
    configured = Path(cwd)
    candidate = configured if configured.is_absolute() else workspace_root / configured
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        return None, f"invalid command cwd {cwd!r}: {exc}"
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return None, f"command cwd escapes workspace: {cwd!r}"
    return resolved, None


def _git_state(workspace: Path) -> dict[str, Any]:
    return {
        "status": _run_git(["git", "status", "--short"], workspace),
        "diff_stat": _run_git(["git", "diff", "--stat"], workspace),
        "diff": _run_git(["git", "diff"], workspace),
        "changed_files": _changed_files(workspace),
    }


def _changed_files(workspace: Path) -> list[str]:
    status = _run_git(["git", "status", "--short"], workspace)
    files: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[2:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    return sorted(set(files))


def _run_git(args: list[str], workspace: Path) -> str:
    try:
        completed = subprocess.run(
            args,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _ensure_git_baseline(workspace: Path) -> None:
    if (workspace / ".git").exists():
        return
    subprocess.run(["git", "init"], cwd=str(workspace), capture_output=True, text=True, check=False)
    subprocess.run(["git", "add", "-A"], cwd=str(workspace), capture_output=True, text=True, check=False)
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "agent-evals",
            "GIT_AUTHOR_EMAIL": "agent-evals@example.invalid",
            "GIT_COMMITTER_NAME": "agent-evals",
            "GIT_COMMITTER_EMAIL": "agent-evals@example.invalid",
        }
    )
    subprocess.run(
        ["git", "commit", "-m", "agent-evals baseline"],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _session_dir(
    case: EvalCase,
    run_id: str,
    workspace: Path,
    pi_config: dict[str, Any],
) -> Path:
    configured = pi_config.get("session_dir")
    if configured:
        return Path(str(configured)).expanduser()
    return workspace.parent / "session_dir" / run_id / case.id


def _pi_env(session_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PI_CODING_AGENT_SESSION_DIR"] = str(session_dir)
    return env


def _task_text(case: EvalCase) -> str:
    return "\n".join(str(message.content) for message in case.input.messages)


def _source_workspace(pi_config: dict[str, Any]) -> Path | None:
    configured = pi_config.get("workspace")
    if not configured:
        return None
    return Path(str(configured)).expanduser()


def _pi_config(case: EvalCase) -> dict[str, Any]:
    metadata_extra = case.metadata.model_extra or {}
    pi_config = metadata_extra.get("pi", {})
    return pi_config if isinstance(pi_config, dict) else {}


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    return {name for name in names if name in ignored}
