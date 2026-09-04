from pathlib import Path

import pytest

from nanoeval.solvers.computer_tasks.code_execution_interface import ExecutionResult
from paperbench.reproduce import run_reproduce_script


class _FakeComputer:
    def __init__(self) -> None:
        self.commands: list[str] = []

    async def check_shell_command(self, cmd: str) -> ExecutionResult:
        self.commands.append(cmd)
        return ExecutionResult(output=b"", exit_code=0)

    async def send_shell_command(self, cmd: str) -> ExecutionResult:
        self.commands.append(cmd)
        if cmd.startswith("cat "):
            return ExecutionResult(output=b"reproduction log", exit_code=0)
        return ExecutionResult(output=b"", exit_code=124)


@pytest.mark.asyncio
async def test_timeout_is_enforced_inside_reproducer_before_returning() -> None:
    computer = _FakeComputer()

    outcome = await run_reproduce_script(
        computer=computer,  # type: ignore[arg-type]
        submission_path=Path("/submission"),
        run_group_id="group",
        runs_dir="/runs",
        run_id="run",
        timeout=5,
    )

    reproduce_command = computer.commands[1]
    assert "timeout --signal=TERM --kill-after=30s 5s bash -c" in reproduce_command
    assert outcome.timedout is True
