from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from nanoeval.solvers.computer_tasks.code_execution_interface import ExecutionResult
from paperbench.computer_utils import put_submission_in_computer


@pytest.mark.asyncio
async def test_submission_archive_uses_a_unique_temporary_path() -> None:
    computer = AsyncMock()
    computer.check_shell_command.return_value = ExecutionResult(
        exit_code=0, output=b""
    )

    with patch(
        "paperbench.computer_utils.put_file_in_computer", new_callable=AsyncMock
    ) as put_file, patch(
        "paperbench.computer_utils.uuid.uuid4"
    ) as uuid4:
        uuid4.return_value.hex = "unique"
        await put_submission_in_computer(
            computer=computer,
            submission_path="/host/submission.tar.gz",
            run_group_id="group",
            runs_dir="/runs",
            run_id="run",
        )

    put_file.assert_awaited_once()
    assert put_file.await_args.kwargs["dest_path"] == (
        "/tmp/pb_submission_unique.tar.gz"
    )
    extraction_command = computer.check_shell_command.await_args_list[0].args[0]
    assert "tar -xzf /tmp/pb_submission_unique.tar.gz" in extraction_command
    assert "rm -f /tmp/pb_submission_unique.tar.gz" in extraction_command
