from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import sys
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
    CODEX_AUTH_CHMOD_COMMAND,
    CODEX_AUTH_PATH,
    CODEX_AUTH_STATUS_COMMAND,
    CODEX_EVENT_LOG,
    CODEX_FORK_CHECKPOINT_CONTAINER_PATH,
    CODEX_FORK_SESSION_ARCHIVE,
    CODEX_FORK_SOURCE_ARCHIVE,
    CODEX_HOME,
    CODEX_HOME_PREPARE_COMMAND,
    CODEX_REVIEW_EVENT_LOG,
    CODEX_ROLLOUT_METADATA,
    INSTRUCTIONS_PATH,
    URL_OBSERVATIONS_FILENAME,
    CodexSolver,
    build_codex_command,
    build_codex_fork_command,
    build_codex_fork_restore_command,
    build_codex_resume_command,
    build_codex_session_checkpoint_command,
    build_codex_user_instructions,
    extract_codex_thread_id,
    extract_url_observations,
)
from paperbench.solvers.completion_review import (
    _COPY_CHANGED_ARTIFACTS_SCRIPT,
    _SNAPSHOT_SCRIPT,
    DEFAULT_MIN_REMAINING_SECONDS,
    EXECUTION_FEEDBACK_MAX_ARTIFACT_BYTES,
    build_completion_review_prompt,
    build_execution_feedback_command,
    build_execution_feedback_prompt,
    execution_feedback_timeout_seconds,
    remaining_budget_seconds,
)


class FakeComputer(ComputerInterface):
    def __init__(
        self,
        *,
        exit_code: int = 0,
        event_log: bytes = b'{"type":"done"}\n',
        review_event_log: bytes = b'{"type":"turn.completed"}\n',
        auth_exit_code: int = 0,
        version_exit_code: int = 0,
        command_delay: float = 0,
        gpu_name: str | None = None,
        submission_heads: list[str] | None = None,
        fork_checkpoint: bytes = b"fork-checkpoint",
    ) -> None:
        self.exit_code = exit_code
        self.event_log = event_log
        self.review_event_log = review_event_log
        self.auth_exit_code = auth_exit_code
        self.version_exit_code = version_exit_code
        self.command_delay = command_delay
        self.gpu_name = gpu_name
        self.submission_heads = list(submission_heads or [])
        self.fork_checkpoint = fork_checkpoint
        self.commands: list[str] = []
        self.uploads: dict[str, bytes] = {}
        self.download_count = 0
        self.refreshed_auth = b'{"tokens": {"access_token": "refreshed"}}\n'

    async def disable_internet(self) -> None:
        raise NotImplementedError

    async def upload(self, file: bytes, destination: str) -> None:
        self.uploads[destination] = file

    async def download(self, file: str) -> bytes:
        if file == CODEX_AUTH_PATH:
            return self.refreshed_auth
        if file == CODEX_FORK_CHECKPOINT_CONTAINER_PATH:
            return self.fork_checkpoint
        if file == CODEX_REVIEW_EVENT_LOG:
            return self.review_event_log
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
        if cmd == "git -C /home/submission rev-parse HEAD":
            if not self.submission_heads:
                return ExecutionResult(output=b"", exit_code=1)
            return ExecutionResult(
                output=(self.submission_heads.pop(0) + "\n").encode(), exit_code=0
            )
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
            evaluation_specification_metadata=lambda: {
                "visible": False,
                "mode": None,
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


def test_completion_review_prompt_is_concise_and_reports_remaining_time() -> None:
    prompt = build_completion_review_prompt(31 * 60 + 59)

    assert prompt == (
        "You have 31 minutes remaining. Continue the current paper-replication task "
        "under the original instructions and use the remaining time to improve and "
        "verify your submission. Do not stop until you have reproduced all results "
        "from the paper. Before finishing, ensure the submission can reproduce all "
        "metrics, figures, tables, and findings from the paper. If all results are "
        "already reproduced, verify the submission against the original instructions "
        "and finish."
    )
    assert DEFAULT_MIN_REMAINING_SECONDS == 3600
    assert remaining_budget_seconds(time_limit_seconds=7200, start_time=100, now=3700) == 3600


def test_execution_feedback_reserves_one_hour_for_review() -> None:
    assert (
        execution_feedback_timeout_seconds(
            remaining_seconds=3 * 3600,
            minimum_review_seconds=3600,
            maximum_execution_seconds=3600,
        )
        == 3600
    )
    assert (
        execution_feedback_timeout_seconds(
            remaining_seconds=90 * 60,
            minimum_review_seconds=3600,
            maximum_execution_seconds=3600,
        )
        == 29 * 60
    )
    assert (
        execution_feedback_timeout_seconds(
            remaining_seconds=3600,
            minimum_review_seconds=3600,
            maximum_execution_seconds=3600,
        )
        == 0
    )


def test_execution_feedback_matches_formal_reproduction_contract() -> None:
    command = build_execution_feedback_command(iteration=2, timeout_seconds=3600)

    assert "--env-file /home/agent.env" in command
    assert "--network bridge" in command
    assert "update-alternatives --set python3 /usr/bin/python3.11" in command
    assert "python3 -m venv venv" in command
    assert command.count("docker create") == 1
    assert "attempt_options=" in command
    assert "600" in command
    assert "reproduce.log" in command
    assert "harness.log" in command
    assert "artifacts" in command
    assert "artifact-manifest" not in command


def test_execution_feedback_copies_only_small_created_or_changed_files(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    destination = tmp_path / "artifacts"
    snapshot = tmp_path / "before.json"
    submission.mkdir()
    (submission / "unchanged.txt").write_text("same")
    (submission / "changed.txt").write_text("before")
    (submission / "link").symlink_to("unchanged.txt")

    subprocess.run(
        [
            sys.executable,
            "-c",
            _SNAPSHOT_SCRIPT,
            str(submission),
            str(snapshot),
            str(EXECUTION_FEEDBACK_MAX_ARTIFACT_BYTES),
        ],
        check=True,
    )
    (submission / "changed.txt").write_text("after")
    (submission / "new.txt").write_text("new")
    (submission / "too-large.bin").write_bytes(
        b"x" * (EXECUTION_FEEDBACK_MAX_ARTIFACT_BYTES + 1)
    )
    (submission / "reproduce.log").write_text("diagnostic log")

    subprocess.run(
        [
            sys.executable,
            "-c",
            _COPY_CHANGED_ARTIFACTS_SCRIPT,
            str(submission),
            str(snapshot),
            str(destination),
            str(EXECUTION_FEEDBACK_MAX_ARTIFACT_BYTES),
        ],
        check=True,
    )

    copied = sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*"))
    assert copied == ["changed.txt", "new.txt"]


def test_execution_feedback_prompt_points_to_reproduction_evidence() -> None:
    prompt = build_execution_feedback_prompt(
        remaining_seconds=3600,
        log_path="/home/logs/execution-feedback/iteration-1/reproduce.log",
        reproduction_timeout_seconds=1800,
    )

    generic_prompt = build_completion_review_prompt(3600)
    common_instruction = generic_prompt.removeprefix("You have 60 minutes remaining. ")

    assert prompt.startswith("You have 60 minutes remaining. ")
    assert "/home/logs/execution-feedback/iteration-1/reproduce.log" in prompt
    assert "/home/logs/execution-feedback/iteration-1/artifacts" in prompt
    assert "exit status" not in prompt
    assert "up to 30 minutes" in prompt
    assert "A diagnostic reproduction was run" in prompt
    assert "Review the execution log" in prompt
    assert prompt.endswith(common_instruction)
    assert "harness.log" not in prompt
    assert "Do not stop until you have reproduced all results from the paper" in prompt
    assert "all metrics, figures, tables, and findings" in prompt


def test_codex_review_resume_uses_exact_thread_and_remaining_budget() -> None:
    event_log = b'noise\n{"type":"thread.started","thread_id":"thread-123"}\n'
    assert extract_codex_thread_id(event_log) == "thread-123"

    command = build_codex_resume_command(
        thread_id="thread-123",
        prompt=build_completion_review_prompt(3600),
        model="gpt-5.6-sol",
        reasoning_effort="high",
        reasoning_summary="detailed",
        time_limit=3600,
    )

    tokens = shlex.split(command)
    assert "resume" in tokens
    assert "thread-123" in tokens
    assert "--ephemeral" not in tokens
    assert "3600s" in tokens
    assert "60 minutes" in command


def test_codex_review_fork_creates_a_persistent_headless_branch() -> None:
    command = build_codex_fork_command(
        parent_thread_id="parent-thread-123",
        prompt="Review this branch.",
        model="gpt-5.6-sol",
        reasoning_effort="high",
        reasoning_summary="detailed",
        time_limit=3600,
    )

    tokens = shlex.split(command)
    assert tokens[tokens.index("exec") + 1] == "fork"
    assert "parent-thread-123" in tokens
    assert "Review this branch." in tokens
    assert "--ephemeral" not in tokens


def test_codex_fork_checkpoint_excludes_auth_and_restores_normalized_inputs() -> None:
    checkpoint = build_codex_session_checkpoint_command()
    restore = build_codex_fork_restore_command()

    assert CODEX_FORK_CHECKPOINT_CONTAINER_PATH in checkpoint
    assert ".codex/sessions" in checkpoint
    assert "auth.json" not in checkpoint
    assert "config.toml" not in checkpoint
    assert CODEX_FORK_SOURCE_ARCHIVE in restore
    assert CODEX_FORK_SESSION_ARCHIVE in restore
    assert "staging/submission" in restore
    assert "--exclude=.codex/auth.json" in restore


@pytest.mark.asyncio
async def test_codex_parent_saves_a_non_secret_fork_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = make_task(tmp_path)
    source_submission = str(tmp_path / "initial-submission.tar.gz")
    Path(source_submission).write_bytes(b"submission")
    monkeypatch.setattr(
        codex_solver_module,
        "snapshot_initial_submission",
        lambda _: source_submission,
    )
    computer = FakeComputer(
        event_log=b'{"type":"thread.started","thread_id":"parent-123"}\n',
    )

    output = await CodexSolver(save_fork_checkpoint=True)._run_agent(computer, task)

    assert output.error_msg is None
    metadata = json.loads((Path(task.run_dir) / "codex-fork-checkpoint.json").read_text())
    assert metadata["parent_thread_id"] == "parent-123"
    assert metadata["source_submission"] == source_submission
    assert metadata["auth_included"] is False
    assert (Path(task.run_dir) / "codex-session-checkpoint.tar.gz").read_bytes() == (
        b"fork-checkpoint"
    )
    checkpoint_commands = [command for command in computer.commands if "tar" in command]
    assert any(".codex/sessions" in command for command in checkpoint_commands)
    assert all("auth.json" not in command for command in checkpoint_commands)


@pytest.mark.asyncio
async def test_codex_child_forks_from_the_saved_parent_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = make_task(tmp_path)
    session_checkpoint = tmp_path / "session.tar.gz"
    source_submission = tmp_path / "submission.tar.gz"
    session_checkpoint.write_bytes(b"session")
    source_submission.write_bytes(b"submission")
    computer = FakeComputer(
        review_event_log=b'{"type":"thread.started","thread_id":"branch-456"}\n',
    )
    remaining = iter([4000, 3599])
    monkeypatch.setattr(
        codex_solver_module,
        "remaining_budget_seconds",
        lambda **_: next(remaining),
    )

    output = await CodexSolver(
        time_limit=7200,
        completion_review=True,
        fork_parent_thread_id="parent-123",
        fork_session_checkpoint_path=str(session_checkpoint),
        fork_source_submission_path=str(source_submission),
    )._run_agent(computer, task)

    assert output.error_msg is None
    fork_commands = [command for command in computer.commands if "codex exec fork" in command]
    assert len(fork_commands) == 1
    assert "parent-123" in fork_commands[0]
    completion = json.loads((Path(task.run_dir) / "completion-review.json").read_text())
    assert completion["forked_from_id"] == "parent-123"
    assert completion["thread_id"] == "branch-456"
    assert completion["performed"] is True
    assert completion["review_count"] == 1
    assert completion["iterations"][0]["operation"] == "fork"


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


def test_codex_solver_keeps_host_auth_out_of_container_mounts(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens": {}}')
    task = cast(PBTask, SimpleNamespace(volumes_config=None))
    solver = CodexSolver(codex_auth_file=str(auth_file), mount_docker_socket=False)

    configured_task = solver._handle_docker_socket_mounting(task)

    assert configured_task.volumes_config is None


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
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens": {"access_token": "initial"}}\n')
    task = make_task(tmp_path)
    computer = FakeComputer()
    solver = CodexSolver(codex_auth_file=str(auth_file))

    await solver._setup_computer(computer, task)

    assert computer.commands == [
        CODEX_HOME_PREPARE_COMMAND,
        CODEX_AUTH_CHMOD_COMMAND,
        "codex --version",
        CODEX_AUTH_STATUS_COMMAND,
    ]
    assert computer.uploads[CODEX_AUTH_PATH] == auth_file.read_bytes()
    assert computer.uploads["/home/logs/codex-version.txt"] == b"codex-cli 0.144.2\n"


@pytest.mark.asyncio
async def test_codex_solver_persists_refreshed_auth_only_when_requested(tmp_path: Path) -> None:
    auth_file = tmp_path / "runtime-codex-home" / "auth.json"
    auth_file.parent.mkdir()
    auth_file.write_text('{"tokens": {"access_token": "initial"}}\n')
    computer = FakeComputer()
    solver = CodexSolver(
        codex_auth_file=str(auth_file),
        persist_refreshed_auth=True,
    )

    await solver._persist_refreshed_auth(computer)

    assert auth_file.read_bytes() == computer.refreshed_auth
    assert auth_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_codex_solver_rejects_agent_image_without_cli(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens": {}}\n')
    task = make_task(tmp_path)
    computer = FakeComputer(version_exit_code=127)
    solver = CodexSolver(codex_auth_file=str(auth_file))

    with pytest.raises(RuntimeError, match="Codex CLI is unavailable"):
        await solver._setup_computer(computer, task)


@pytest.mark.asyncio
async def test_codex_solver_rejects_invalid_chatgpt_auth(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text('{"tokens": {}}\n')
    task = make_task(tmp_path)
    computer = FakeComputer(auth_exit_code=1)
    solver = CodexSolver(codex_auth_file=str(auth_file))

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
async def test_codex_solver_repeats_review_while_at_least_one_hour_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stub_heavy_uploads: list[dict[str, object]],
) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(
        event_log=b'{"type":"thread.started","thread_id":"thread-123"}\n',
    )
    remaining = iter([5000, 4000, 3599])
    monkeypatch.setattr(
        codex_solver_module,
        "remaining_budget_seconds",
        lambda **_: next(remaining),
    )
    solver = CodexSolver(time_limit=7200, completion_review=True)

    output = await solver._run_agent(computer, task)

    assert output.error_msg is None
    agent_commands = [command for command in computer.commands if "codex exec" in command]
    assert len(agent_commands) == 3
    assert "--ephemeral" not in agent_commands[0]
    assert "resume" in shlex.split(agent_commands[1])
    assert "thread-123" in shlex.split(agent_commands[1])
    assert "resume" in shlex.split(agent_commands[2])
    assert len(stub_heavy_uploads) == 2
    completion = json.loads((Path(task.run_dir) / "completion-review.json").read_text())
    assert completion["performed"] is True
    assert completion["review_count"] == 2
    assert completion["remaining_seconds_at_start"] == 5000
    assert [item["remaining_seconds_at_start"] for item in completion["iterations"]] == [
        5000,
        4000,
    ]
    assert completion["completion_reason"] == "insufficient-original-budget"
    assert "review_seconds" not in completion
    assert "environment_variant" not in completion


@pytest.mark.asyncio
async def test_codex_solver_counts_execution_feedback_against_shared_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(
        event_log=b'{"type":"thread.started","thread_id":"thread-123"}\n',
    )
    remaining = iter([7200, 3600, 3599])
    monkeypatch.setattr(
        codex_solver_module,
        "remaining_budget_seconds",
        lambda **_: next(remaining),
    )

    output = await CodexSolver(
        time_limit=24 * 3600,
        completion_review=True,
        completion_review_mode="execution-log",
        execution_feedback_timeout_seconds=3600,
    )._run_agent(computer, task)

    assert output.error_msg is None
    diagnostic_commands = [
        command for command in computer.commands if "pb-execution-feedback-" in command
    ]
    assert len(diagnostic_commands) == 1
    assert "diagnostic_budget_seconds=3540" in diagnostic_commands[0]
    assert "pb-reproducer:latest" in diagnostic_commands[0]
    assert "docker cp /home/submission/." in diagnostic_commands[0]
    assert "--shm-size 8g" in diagnostic_commands[0]
    assert 'NVIDIA_VISIBLE_DEVICES}" == all' in diagnostic_commands[0]
    assert "gpu_args=(--gpus all)" in diagnostic_commands[0]
    review_commands = [command for command in computer.commands if "codex exec resume" in command]
    assert len(review_commands) == 1
    assert "/home/logs/execution-feedback/iteration-1/reproduce.log" in review_commands[0]
    completion = json.loads((Path(task.run_dir) / "completion-review.json").read_text())
    assert completion["mode"] == "execution-log"
    assert completion["budget_policy"] == "shared-agent-and-diagnostic-wall-clock"
    assert completion["review_count"] == 1
    assert completion["iterations"][0]["execution_feedback"]["timeout_seconds"] == 3540
    assert completion["completion_reason"] == "insufficient-original-budget"


@pytest.mark.asyncio
async def test_codex_solver_stops_after_two_quick_unchanged_reviews(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(
        event_log=b'{"type":"thread.started","thread_id":"thread-123"}\n',
        submission_heads=["a" * 40] * 4,
    )
    remaining = iter([5000, 4900, 4800, 3599])
    monkeypatch.setattr(
        codex_solver_module,
        "remaining_budget_seconds",
        lambda **_: next(remaining),
    )

    output = await CodexSolver(time_limit=7200, completion_review=True)._run_agent(computer, task)

    assert output.error_msg is None
    agent_commands = [command for command in computer.commands if "codex exec" in command]
    assert len(agent_commands) == 3
    completion = json.loads((Path(task.run_dir) / "completion-review.json").read_text())
    assert completion["review_count"] == 2
    assert completion["completion_reason"] == "consecutive-quick-unchanged"
    assert completion["consecutive_quick_unchanged"] == 2
    assert all(item["quick_unchanged"] is True for item in completion["iterations"])


@pytest.mark.asyncio
async def test_codex_solver_skips_review_below_one_hour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = make_task(tmp_path)
    computer = FakeComputer(
        event_log=b'{"type":"thread.started","thread_id":"thread-123"}\n',
    )
    monkeypatch.setattr(codex_solver_module, "remaining_budget_seconds", lambda **_: 3599)

    await CodexSolver(time_limit=7200, completion_review=True)._run_agent(computer, task)

    agent_commands = [command for command in computer.commands if "codex exec" in command]
    assert len(agent_commands) == 1
    completion = json.loads((Path(task.run_dir) / "completion-review.json").read_text())
    assert completion["performed"] is False
    assert completion["skip_reason"] == "insufficient-original-budget"


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
