from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import blobfile as bf
import pytest
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerRuntimeNoJupyter

import paperbench.solvers.codex.solver as codex_solver_module
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
)
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.basicagent.prompts.templates import (
    REPRODUCE_SH_MESSAGE,
    get_system_message,
)
from paperbench.solvers.codex.solver import (
    CODEX_AUTH_PATH,
    CODEX_AUTH_STATUS_COMMAND,
    CODEX_EVENT_LOG,
    CODEX_HOME,
    CODEX_ROLLOUT_METADATA,
    INSTRUCTIONS_PATH,
    URL_OBSERVATIONS_FILENAME,
    CodexSolver,
    build_codex_command,
    build_codex_user_instructions,
    extract_url_observations,
)


class FakeComputer(ComputerInterface):
    def __init__(
        self,
        *,
        exit_code: int = 0,
        event_log: bytes = b'{"type":"done"}\n',
        auth_exit_code: int = 0,
        version_exit_code: int = 0,
        command_delay: float = 0,
        gpu_name: str | None = None,
    ) -> None:
        self.exit_code = exit_code
        self.event_log = event_log
        self.auth_exit_code = auth_exit_code
        self.version_exit_code = version_exit_code
        self.command_delay = command_delay
        self.gpu_name = gpu_name
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.download_count = 0

    async def disable_internet(self) -> None:
        raise NotImplementedError

    async def upload(self, file: bytes, destination: str) -> None:
        self.uploads[destination] = file

    async def download(self, file: str) -> bytes:
        assert file == CODEX_EVENT_LOG
        self.download_count += 1
        return self.event_log

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        self.commands.append(cmd)
        if cmd == "codex --version":
            return ExecutionResult(
                output=b"codex-cli 0.144.2\n",
                exit_code=self.version_exit_code,
            )
        if cmd == CODEX_AUTH_STATUS_COMMAND:
            return ExecutionResult(
                output=b"Logged in using ChatGPT\n",
                exit_code=self.auth_exit_code,
            )
        if cmd == "nvidia-smi --query-gpu=name --format=csv,noheader":
            output = f"{self.gpu_name}\n".encode() if self.gpu_name else b""
            return ExecutionResult(output=output, exit_code=0 if self.gpu_name else 1)
        await asyncio.sleep(self.command_delay)
        return ExecutionResult(output=b"", exit_code=self.exit_code)

    async def fetch_container_names(self) -> list[str]:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError


def make_task(tmp_path: Path) -> PBTask:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return cast(
        PBTask,
        SimpleNamespace(
            run_id="semantic-self-consistency_test",
            run_group_id="group",
            runs_dir=str(tmp_path),
            run_dir=str(run_dir),
            requirements_input_metadata=lambda: {
                "provided": True,
                "sha256": "abc123",
                "requirement_count": 2,
                "container_path": "/home/requirements/requirements.csv",
            },
            prompt=[{"role": "user", "content": "BASE PAPERBENCH TASK\n"}],
            paper_id="semantic-self-consistency",
            judge=SimpleNamespace(code_only=True),
        ),
    )


@pytest.fixture(autouse=True)
def stub_heavy_uploads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    uploads: list[dict[str, object]] = []

    async def record_upload(**kwargs: object) -> None:
        uploads.append(kwargs)

    monkeypatch.setattr(codex_solver_module, "upload_heavy_logs", record_upload)
    return uploads


def test_build_codex_user_instructions_matches_basicagent_runtime_context() -> None:
    instructions = build_codex_user_instructions(
        base_instructions=(
            "BASE PAPERBENCH TASK\nAn additional list is at /home/requirements/requirements.csv.\n"
        ),
        time_limit=24 * 60 * 60,
        gpu_generation="NVIDIA A100-SXM4-80GB",
    )

    assert "BASE PAPERBENCH TASK" in instructions
    assert "/home/requirements/requirements.csv" in instructions
    assert "24.0 hours" in instructions
    assert "NVIDIA A100-SXM4-80GB" in instructions
    assert "/home/agent.env" in instructions
    assert "root access" in instructions
    assert get_system_message(iterative=False, code_only=True) not in instructions
    assert REPRODUCE_SH_MESSAGE not in instructions


def test_build_codex_user_instructions_reports_missing_gpu() -> None:
    instructions = build_codex_user_instructions(
        base_instructions="BASE PAPERBENCH TASK\n",
        time_limit=60,
        gpu_generation=None,
    )

    assert "You don't have access to a GPU" in instructions
    assert "0.017 hours" in instructions


def test_build_codex_command_pins_reproducibility_controls() -> None:
    developer_instructions = get_system_message(iterative=False, code_only=True)
    command = build_codex_command(
        model="gpt-5.6-sol",
        reasoning_effort="high",
        reasoning_summary="detailed",
        time_limit=3600,
        developer_instructions=developer_instructions,
    )

    assert command.startswith(
        f"env -u OPENAI_API_KEY CODEX_HOME={CODEX_HOME} "
        "timeout --signal=TERM --kill-after=30s 3600s"
    )
    assert "codex exec" in command
    assert "--model gpt-5.6-sol" in command
    assert 'model_reasoning_effort="high"' in command
    assert 'model_reasoning_summary="detailed"' in command
    config_values = [
        value
        for index, value in enumerate(shlex.split(command))
        if index > 0 and shlex.split(command)[index - 1] == "-c"
    ]
    assert f"developer_instructions={json.dumps(developer_instructions)}" in config_values
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--skip-git-repo-check" in command
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    assert "< /home/instructions.txt" in command
    assert f"> {CODEX_EVENT_LOG} 2>&1" in command


@pytest.mark.asyncio
async def test_run_metadata_records_requirements_condition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_run_agent(
        self: CodexSolver, computer: ComputerInterface, task: PBTask
    ) -> AgentOutput:
        del self, computer
        return AgentOutput(
            run_id=task.run_id,
            time_start=1.0,
            time_end=2.0,
            runtime_in_seconds=1.0,
            status_exists=True,
        )

    monkeypatch.setattr(CodexSolver, "_run_agent", fake_run_agent)
    task = make_task(tmp_path)

    await CodexSolver()._run_save_and_check(FakeComputer(), task)

    metadata = json.loads((Path(task.run_dir) / "metadata.json").read_text())
    assert metadata["requirements_input"] == {
        "provided": True,
        "sha256": "abc123",
        "requirement_count": 2,
        "container_path": "/home/requirements/requirements.csv",
    }


def test_extract_url_observations_from_web_search_and_commands() -> None:
    event_log = b"\n".join(
        [
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "search-1",
                        "type": "web_search",
                        "query": "official dataset",
                        "action": {
                            "type": "openPage",
                            "url": "https://example.com/data?a=1",
                        },
                    },
                }
            ).encode(),
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "cmd-1",
                        "type": "command_execution",
                        "command": "curl 'https://api.example.com/file?token=secret&part=1'",
                        "status": "in_progress",
                    },
                }
            ).encode(),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "cmd-1",
                        "type": "command_execution",
                        "command": "curl 'https://api.example.com/file?token=secret&part=1'",
                        "status": "completed",
                    },
                }
            ).encode(),
        ]
    )

    observations = [json.loads(line) for line in extract_url_observations(event_log).splitlines()]

    assert observations == [
        {
            "source": "web_search",
            "url": "https://example.com/data?a=1",
            "event_type": "item.completed",
            "item_id": "search-1",
            "web_action": "openPage",
        },
        {
            "source": "command_text",
            "url": "https://api.example.com/file?token=REDACTED&part=1",
            "event_type": "item.completed",
            "item_id": "cmd-1",
            "command": "curl 'https://api.example.com/file?token=REDACTED&part=1'",
        },
    ]


def test_extract_url_observations_skips_invalid_json_and_redacts_userinfo() -> None:
    event_log = b"\n".join(
        [
            b"not json",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "cmd-2",
                        "type": "command_execution",
                        "command": "wget https://user:password@example.net/archive.zip",
                    },
                }
            ).encode(),
        ]
    )

    observations = [json.loads(line) for line in extract_url_observations(event_log).splitlines()]

    assert observations[0]["url"] == "https://REDACTED@example.net/archive.zip"
    assert observations[0]["command"] == "wget https://REDACTED@example.net/archive.zip"


def test_codex_solver_uses_shell_only_runtime() -> None:
    assert isinstance(CodexSolver().computer_runtime, AlcatrazComputerRuntimeNoJupyter)


def test_codex_solver_mounts_chatgpt_auth_read_only(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens": {}}')
    task = cast(PBTask, SimpleNamespace(volumes_config=None))
    solver = CodexSolver(codex_auth_file=str(auth_file), mount_docker_socket=False)

    configured_task = solver._handle_docker_socket_mounting(task)

    assert configured_task.volumes_config == {
        "codexauth": {
            "bind_source": str(auth_file.resolve()),
            "bind_dest": CODEX_AUTH_PATH,
            "mode": "ro",
        }
    }


def test_codex_solver_rejects_missing_chatgpt_auth_file(tmp_path: Path) -> None:
    task = cast(PBTask, SimpleNamespace(volumes_config=None))
    solver = CodexSolver(
        codex_auth_file=str(tmp_path / "missing.json"),
        mount_docker_socket=False,
    )

    with pytest.raises(FileNotFoundError, match="Codex ChatGPT auth file not found"):
        solver._handle_docker_socket_mounting(task)


@pytest.mark.asyncio
async def test_codex_solver_checks_cli_and_records_version(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer()
    solver = CodexSolver()

    await solver._setup_computer(computer, task)

    assert computer.commands == ["codex --version", CODEX_AUTH_STATUS_COMMAND]
    assert computer.uploads["/home/logs/codex-version.txt"] == b"codex-cli 0.144.2\n"


@pytest.mark.asyncio
async def test_codex_solver_rejects_agent_image_without_cli(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(version_exit_code=127)
    solver = CodexSolver()

    with pytest.raises(RuntimeError, match="Codex CLI is unavailable"):
        await solver._setup_computer(computer, task)


@pytest.mark.asyncio
async def test_codex_solver_rejects_invalid_chatgpt_auth(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(auth_exit_code=1)
    solver = CodexSolver()

    with pytest.raises(RuntimeError, match="ChatGPT subscription authentication"):
        await solver._setup_computer(computer, task)


@pytest.mark.asyncio
async def test_codex_solver_records_log_status_and_runtime(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(
        event_log=b'{"command":"git status"}\n',
        gpu_name="NVIDIA A100-SXM4-80GB",
    )
    solver = CodexSolver(
        model="gpt-5.6-sol",
        reasoning_effort="high",
        reasoning_summary="detailed",
        time_limit=60,
    )

    output = await solver._run_agent(computer, task)

    assert isinstance(output, AgentOutput)
    assert output.error_msg is None
    assert output.status_exists is True
    assert output.runtime_in_seconds >= 0
    assert computer.commands[:4] == [
        "nvidia-smi --query-gpu=name --format=csv,noheader",
        "docker --version",
        "docker run --rm hello-world",
        "docker ps -a",
    ]
    assert len(computer.commands) == 5
    assert "codex exec" in computer.commands[-1]
    uploaded_instructions = computer.uploads[INSTRUCTIONS_PATH].decode()
    assert "BASE PAPERBENCH TASK" in uploaded_instructions
    assert "NVIDIA A100-SXM4-80GB" in uploaded_instructions
    assert "0.017 hours" in uploaded_instructions
    developer_instructions = get_system_message(iterative=False, code_only=True)
    assert developer_instructions not in uploaded_instructions
    assert f"developer_instructions={json.dumps(developer_instructions)}" in shlex.split(
        computer.commands[-1]
    )
    with bf.BlobFile(bf.join(task.run_dir, "agent.log"), "rb") as handle:
        assert handle.read() == b'{"command":"git status"}\n'
    with bf.BlobFile(bf.join(task.run_dir, URL_OBSERVATIONS_FILENAME), "rb") as handle:
        assert handle.read() == b""
    metadata = json.loads(computer.uploads[CODEX_ROLLOUT_METADATA])
    assert metadata["model"] == "gpt-5.6-sol"
    assert metadata["reasoning_effort"] == "high"
    assert metadata["reasoning_summary"] == "detailed"
    assert metadata["time_limit_seconds"] == 60
    assert metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_codex_solver_uploads_final_submission_checkpoint(
    tmp_path: Path,
    stub_heavy_uploads: list[dict[str, object]],
) -> None:
    task = make_task(tmp_path)
    solver = CodexSolver(time_limit=60)

    output = await solver._run_agent(FakeComputer(), task)

    assert output.error_msg is None
    assert len(stub_heavy_uploads) == 1
    assert stub_heavy_uploads[0]["run_id"] == task.run_id
    assert stub_heavy_uploads[0]["runtime"] == pytest.approx(output.runtime_in_seconds)


@pytest.mark.asyncio
async def test_codex_solver_preserves_partial_submission_on_timeout(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(exit_code=124)
    solver = CodexSolver(time_limit=1)

    output = await solver._run_agent(computer, task)

    assert output.error_msg == "Codex rollout timed out after 1 seconds"
    assert output.status_exists is True
    assert Path(task.run_dir, "status.json").exists()


@pytest.mark.asyncio
async def test_codex_solver_records_nonzero_exit_without_raising(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(exit_code=2)
    solver = CodexSolver(time_limit=60)

    output = await solver._run_agent(computer, task)

    assert output.error_msg == "Codex rollout exited with status 2"
    assert output.status_exists is True


@pytest.mark.asyncio
async def test_codex_solver_checkpoints_submission_during_long_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(command_delay=0.03)
    checkpoints: list[float | None] = []

    async def record_checkpoint(**kwargs: object) -> None:
        runtime = kwargs["runtime"]
        assert runtime is None or isinstance(runtime, float)
        checkpoints.append(runtime)

    monkeypatch.setattr(codex_solver_module, "upload_heavy_logs", record_checkpoint)
    solver = CodexSolver(time_limit=60, upload_interval_seconds=0.005)

    output = await solver._run_agent(computer, task)

    assert output.error_msg is None
    assert checkpoints
    assert all(runtime is not None and runtime > 0 for runtime in checkpoints)
    assert computer.download_count > 1
