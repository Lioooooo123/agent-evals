from __future__ import annotations

import subprocess
import sys


def test_run_help_outputs_usage():
    result = subprocess.run(
        [sys.executable, "-m", "agent_evals.cli", "run", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "agent-evals run" in result.stdout
