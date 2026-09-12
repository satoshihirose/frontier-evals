from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import blobfile as bf
import structlog.stdlib
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerRuntimeNoJupyter
from typing_extensions import override

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
)
from paperbench.constants import AGENT_DIR_CONFIG, LOGS_DIR, SUBMISSION_DIR, WORKSPACE_BASE
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver
from paperbench.solvers.basicagent.prompts.templates import get_system_message
from paperbench.solvers.basicagent.utils import (
    build_user_instructions,
    get_gpu_generation,
    get_task_instruction_text,
)
from paperbench.solvers.completion_review import (
    DEFAULT_MIN_REMAINING_SECONDS,
    EXECUTION_FEEDBACK_INPUT_VERSION,
    MAX_CONSECUTIVE_QUICK_UNCHANGED,
    QUICK_UNCHANGED_MAX_SECONDS,
    CompletionReviewMode,
    build_completion_review_prompt,
    build_execution_feedback_command,
    build_execution_feedback_prompt,
    execution_feedback_paths,
    execution_feedback_timeout_seconds,
    get_submission_git_head,
    is_execution_feedback_infrastructure_error,
    is_quick_unchanged_review,
    remaining_budget_seconds,
    snapshot_initial_submission,
    write_completion_review_metadata,
)
from paperbench.solvers.upload import upload_heavy_logs, upload_status
from paperbench.solvers.utils import check_for_existing_run, sanity_check_docker

logger = structlog.stdlib.get_logger(component=__name__)

CODEX_EVENT_LOG = f"{LOGS_DIR}/codex-events.jsonl"
CODEX_REVIEW_EVENT_LOG = f"{LOGS_DIR}/codex-review-events.jsonl"
CODEX_ROLLOUT_METADATA = f"{LOGS_DIR}/codex-rollout.json"
CODEX_FORK_CHECKPOINT_CONTAINER_PATH = f"{LOGS_DIR}/codex-session-checkpoint.tar.gz"
CODEX_FORK_CHECKPOINT_FILENAME = "codex-session-checkpoint.tar.gz"
CODEX_FORK_METADATA_FILENAME = "codex-fork-checkpoint.json"
CODEX_FORK_SOURCE_ARCHIVE = "/tmp/codex-fork-source-submission.tar.gz"
CODEX_FORK_SESSION_ARCHIVE = "/tmp/codex-fork-session-checkpoint.tar.gz"
URL_OBSERVATIONS_FILENAME = "url-observations.jsonl"
CODEX_HOME = f"{WORKSPACE_BASE}/.codex"
CODEX_AUTH_PATH = f"{CODEX_HOME}/auth.json"
CODEX_HOME_PREPARE_COMMAND = f"install -d -m 700 {CODEX_HOME}"
CODEX_AUTH_CHMOD_COMMAND = f"chmod 600 {CODEX_AUTH_PATH}"
CODEX_AUTH_STATUS_COMMAND = f"env -u OPENAI_API_KEY CODEX_HOME={CODEX_HOME} codex login status"
INSTRUCTIONS_PATH = f"{WORKSPACE_BASE}/instructions.txt"
ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
ReasoningSummary = Literal["auto", "concise", "detailed", "none"]
URL_PATTERN = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
SENSITIVE_QUERY_KEY = re.compile(
    r"(?:auth|code|credential|key|passw|secret|signature|token)",
    re.IGNORECASE,
)


def build_codex_user_instructions(
    *,
    base_instructions: str,
    time_limit: int,
    gpu_generation: str | None,
) -> str:
    """Add the user-level runtime context that BasicAgent receives."""
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    return build_user_instructions(
        base_instructions=base_instructions,
        time_limit=time_limit,
        gpu_generation=gpu_generation,
    )


def _redact_url(url: str) -> str:
    """Remove credentials and sensitive query values from a logged URL."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port is not None else ""
        userinfo = "REDACTED@" if parts.username is not None else ""
        netloc = f"{userinfo}{host}{port}"
        query = urlencode(
            [
                (key, "REDACTED" if SENSITIVE_QUERY_KEY.search(key) else value)
                for key, value in parse_qsl(parts.query, keep_blank_values=True)
            ],
            doseq=True,
        )
        return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))
    except ValueError:
        return url


def _find_urls(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for match in URL_PATTERN.finditer(text):
        raw_url = match.group(0).rstrip(".,;:!?)]}")
        if raw_url:
            found.append((raw_url, _redact_url(raw_url)))
    return found


def _iter_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _iter_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _iter_strings(item)]
    return []


def extract_url_observations(event_log: bytes) -> bytes:
    """Extract redacted URLs exposed by Codex web-search and command events."""
    observations: dict[tuple[str, str, str], dict[str, str]] = {}
    for line_number, raw_line in enumerate(event_log.splitlines(), start=1):
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict) or not isinstance(event.get("item"), dict):
            continue

        item = event["item"]
        item_type = item.get("type")
        event_type = str(event.get("type", "unknown"))
        item_id = str(item.get("id", f"line-{line_number}"))

        if item_type == "web_search":
            action = item.get("action")
            action_type = (
                str(action.get("type", "unknown")) if isinstance(action, dict) else "unknown"
            )
            for text in _iter_strings(item):
                for _, url in _find_urls(text):
                    key = ("web_search", item_id, url)
                    observations[key] = {
                        "source": "web_search",
                        "url": url,
                        "event_type": event_type,
                        "item_id": item_id,
                        "web_action": action_type,
                    }
        elif item_type == "command_execution" and isinstance(item.get("command"), str):
            command = item["command"]
            urls = _find_urls(command)
            if not urls:
                continue
            redacted_command = command
            for raw_url, url in urls:
                redacted_command = redacted_command.replace(raw_url, url)
            for _, url in urls:
                key = ("command_text", item_id, url)
                observations[key] = {
                    "source": "command_text",
                    "url": url,
                    "event_type": event_type,
                    "item_id": item_id,
                    "command": redacted_command,
                }

    if not observations:
        return b""
    return (
        "\n".join(json.dumps(record, ensure_ascii=False) for record in observations.values()) + "\n"
    ).encode()


def build_codex_command(
    *,
    model: str,
    reasoning_effort: ReasoningEffort,
    reasoning_summary: ReasoningSummary,
    time_limit: int,
    developer_instructions: str,
    ephemeral: bool = True,
) -> str:
    """Build the non-interactive Codex command executed in the agent container."""
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")

    args = [
        "env",
        "-u",
        "OPENAI_API_KEY",
        f"CODEX_HOME={CODEX_HOME}",
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{time_limit}s",
        "codex",
        "exec",
        "-C",
        WORKSPACE_BASE,
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        f'model_reasoning_summary="{reasoning_summary}"',
        "-c",
        f"developer_instructions={json.dumps(developer_instructions)}",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-",
    ]
    if ephemeral:
        args.insert(args.index("--ignore-user-config"), "--ephemeral")
    return (
        f"{shlex.join(args)} < {shlex.quote(INSTRUCTIONS_PATH)} "
        f"> {shlex.quote(CODEX_EVENT_LOG)} 2>&1"
    )


def extract_codex_thread_id(event_log: bytes) -> str | None:
    for raw_line in event_log.splitlines():
        try:
            event = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str) and thread_id:
            return thread_id
    return None


def build_codex_resume_command(
    *,
    thread_id: str,
    prompt: str,
    model: str,
    reasoning_effort: ReasoningEffort,
    reasoning_summary: ReasoningSummary,
    time_limit: int,
) -> str:
    if not thread_id:
        raise ValueError("thread_id must be non-empty")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    args = [
        "env",
        "-u",
        "OPENAI_API_KEY",
        f"CODEX_HOME={CODEX_HOME}",
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{time_limit}s",
        "codex",
        "exec",
        "resume",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        f'model_reasoning_summary="{reasoning_summary}"',
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        thread_id,
        prompt,
    ]
    return f"{shlex.join(args)} > {shlex.quote(CODEX_REVIEW_EVENT_LOG)} 2>&1"


def build_codex_fork_command(
    *,
    parent_thread_id: str,
    prompt: str,
    model: str,
    reasoning_effort: ReasoningEffort,
    reasoning_summary: ReasoningSummary,
    time_limit: int,
) -> str:
    """Fork a saved Codex session into a persistent non-interactive branch."""
    if not parent_thread_id:
        raise ValueError("parent_thread_id must be non-empty")
    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    args = [
        "env",
        "-u",
        "OPENAI_API_KEY",
        f"CODEX_HOME={CODEX_HOME}",
        "timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{time_limit}s",
        "codex",
        "exec",
        "fork",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        f'model_reasoning_summary="{reasoning_summary}"',
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        parent_thread_id,
        prompt,
    ]
    return f"{shlex.join(args)} > {shlex.quote(CODEX_REVIEW_EVENT_LOG)} 2>&1"


def build_codex_session_checkpoint_command() -> str:
    """Archive resumable Codex state without authentication material."""
    script = "\n".join(
        [
            "set -euo pipefail",
            "paths=(.codex/sessions)",
            (
                "for path in .codex/session_index.jsonl .codex/state_5.sqlite "
                ".codex/state_5.sqlite-shm .codex/state_5.sqlite-wal; do"
            ),
            '  [[ ! -e "$path" ]] || paths+=("$path")',
            "done",
            f"tar -czf {shlex.quote(CODEX_FORK_CHECKPOINT_CONTAINER_PATH)} "
            f'-C {WORKSPACE_BASE} "${{paths[@]}}"',
        ]
    )
    return f"bash -lc {shlex.quote(script)}"


def build_codex_fork_restore_command() -> str:
    """Restore the normalized parent submission and non-secret session state."""
    script = "\n".join(
        [
            "set -euo pipefail",
            f"staging=$(mktemp -d {WORKSPACE_BASE}/.fork-source.XXXXXX)",
            'cleanup() { rm -rf -- "$staging"; }',
            "trap cleanup EXIT",
            f'tar -xzf {shlex.quote(CODEX_FORK_SOURCE_ARCHIVE)} -C "$staging"',
            'test -d "$staging/submission"',
            f"rm -rf -- {shlex.quote(SUBMISSION_DIR)}",
            f'mv "$staging/submission" {shlex.quote(SUBMISSION_DIR)}',
            (
                "tar --exclude=.codex/auth.json --exclude=.codex/config.toml "
                f"-xzf {shlex.quote(CODEX_FORK_SESSION_ARCHIVE)} -C {WORKSPACE_BASE}"
            ),
        ]
    )
    return f"bash -lc {shlex.quote(script)}"


@chz.chz
class CodexSolver(BasePBSolver):
    """Run Codex CLI inside the standard PaperBench agent computer."""

    computer_runtime: AlcatrazComputerRuntimeNoJupyter = chz.field(
        default_factory=AlcatrazComputerRuntimeNoJupyter
    )
    model: str = chz.field(default="gpt-5.6-sol")
    reasoning_effort: ReasoningEffort = chz.field(default="high")
    reasoning_summary: ReasoningSummary = chz.field(
        default="detailed",
        doc="Public reasoning summary detail emitted into the Codex JSONL event stream",
    )
    codex_auth_file: str = chz.field(
        default="~/.codex/auth.json",
        doc="Host ChatGPT auth file copied into the isolated Codex agent container",
    )
    persist_refreshed_auth: bool = chz.field(
        default=False,
        doc=(
            "Atomically copy the container's refreshed auth state back to codex_auth_file. "
            "Use only with a run-specific auth copy, never the host's primary auth file."
        ),
    )
    time_limit: int = chz.field(default=24 * 60 * 60, doc="Rollout time limit in seconds")
    upload_interval_seconds: float | None = chz.field(
        default=1800,
        doc="Seconds between submission checkpoints; set to null to disable",
    )
    completion_review: bool = chz.field(
        default=False,
        doc="Resume completed rollouts while enough original budget remains",
    )
    completion_review_min_remaining_seconds: int = chz.field(
        default=DEFAULT_MIN_REMAINING_SECONDS,
        doc="Minimum original rollout budget required before starting the review",
    )
    completion_review_mode: CompletionReviewMode = chz.field(
        default="generic",
        doc="Use a neutral review or first expose a harness-run reproduce.sh log",
    )
    execution_feedback_timeout_seconds: int = chz.field(
        default=60 * 60,
        doc="Maximum reproduce.sh runtime before each execution-log review",
    )
    save_fork_checkpoint: bool = chz.field(
        default=False,
        doc="Save non-secret Codex session state for later paired review branches",
    )
    fork_parent_thread_id: str | None = chz.field(default=None)
    fork_session_checkpoint_path: str | None = chz.field(default=None)
    fork_source_submission_path: str | None = chz.field(default=None)

    @override
    def shortname(self) -> str:
        return "codex"

    @override
    def _handle_docker_socket_mounting(self, task: PBTask) -> PBTask:
        task = super()._handle_docker_socket_mounting(task)
        auth_file = Path(self.codex_auth_file).expanduser().resolve()
        if not auth_file.is_file():
            raise FileNotFoundError(f"Codex ChatGPT auth file not found: {auth_file}")
        return task

    @override
    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        del task
        auth_file = Path(self.codex_auth_file).expanduser().resolve()
        auth_bytes = auth_file.read_bytes()
        try:
            parsed_auth = json.loads(auth_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Codex ChatGPT auth file is invalid JSON: {auth_file}") from exc
        if not isinstance(parsed_auth, dict):
            raise ValueError(f"Codex ChatGPT auth file must contain a JSON object: {auth_file}")

        prepare_home = await computer.send_shell_command(
            CODEX_HOME_PREPARE_COMMAND,
            idempotent=True,
        )
        if prepare_home.exit_code != 0:
            raise RuntimeError("Could not prepare the isolated Codex home directory")
        await computer.upload(auth_bytes, CODEX_AUTH_PATH)
        protect_auth = await computer.send_shell_command(
            CODEX_AUTH_CHMOD_COMMAND,
            idempotent=True,
        )
        if protect_auth.exit_code != 0:
            raise RuntimeError("Could not protect the isolated Codex authentication file")

        result = await computer.send_shell_command("codex --version", idempotent=True)
        if result.exit_code != 0:
            raise RuntimeError(
                "Codex CLI is unavailable in the agent image: "
                f"{result.output.decode(errors='replace')}"
            )
        await computer.upload(result.output, f"{LOGS_DIR}/codex-version.txt")

        auth_status = await computer.send_shell_command(
            CODEX_AUTH_STATUS_COMMAND,
            idempotent=True,
        )
        auth_output = auth_status.output.decode(errors="replace")
        if auth_status.exit_code != 0 or "chatgpt" not in auth_output.lower():
            raise RuntimeError(
                "Codex ChatGPT subscription authentication is unavailable in the agent "
                "container; the copied auth file must come from `codex login` with ChatGPT"
            )

    async def _persist_refreshed_auth(self, computer: ComputerInterface) -> None:
        if not self.persist_refreshed_auth:
            return

        refreshed_auth = await computer.download(CODEX_AUTH_PATH)
        try:
            parsed_auth = json.loads(refreshed_auth)
        except json.JSONDecodeError as exc:
            raise ValueError("Codex returned an invalid refreshed authentication file") from exc
        if not isinstance(parsed_auth, dict):
            raise ValueError("Codex refreshed authentication must contain a JSON object")

        auth_file = Path(self.codex_auth_file).expanduser().resolve()
        auth_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        file_descriptor, temporary_path = tempfile.mkstemp(
            dir=auth_file.parent,
            prefix=".auth.json.",
            suffix=".tmp",
        )
        try:
            os.write(file_descriptor, refreshed_auth)
            os.fchmod(file_descriptor, 0o600)
            os.close(file_descriptor)
            file_descriptor = -1
            os.replace(temporary_path, auth_file)
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    async def _read_event_log(
        self,
        computer: ComputerInterface,
        fallback: bytes,
        path: str = CODEX_EVENT_LOG,
    ) -> bytes:
        try:
            return await computer.download(path)
        except Exception as exc:
            logger.exception(f"Could not download Codex event log: {exc}")
            return fallback or f"Codex event log unavailable: {exc}\n".encode()

    @staticmethod
    def _write_run_logs(task: PBTask, event_log: bytes) -> None:
        bf.write_bytes(bf.join(task.run_dir, "agent.log"), event_log)
        bf.write_bytes(
            bf.join(task.run_dir, URL_OBSERVATIONS_FILENAME),
            extract_url_observations(event_log),
        )

    async def _restore_fork_checkpoint(self, computer: ComputerInterface) -> None:
        if self.fork_session_checkpoint_path is None:
            raise ValueError("fork_session_checkpoint_path is required")
        if self.fork_source_submission_path is None:
            raise ValueError("fork_source_submission_path is required")
        await computer.upload(
            bf.read_bytes(self.fork_session_checkpoint_path),
            CODEX_FORK_SESSION_ARCHIVE,
        )
        await computer.upload(
            bf.read_bytes(self.fork_source_submission_path),
            CODEX_FORK_SOURCE_ARCHIVE,
        )
        result = await computer.send_shell_command(build_codex_fork_restore_command())
        if result.exit_code != 0:
            raise RuntimeError(
                "Could not restore Codex fork checkpoint: " + result.output.decode(errors="replace")
            )

    async def _save_fork_checkpoint(
        self,
        *,
        computer: ComputerInterface,
        task: PBTask,
        parent_thread_id: str,
        source_submission: str,
        remaining_seconds: int,
    ) -> dict[str, object]:
        result = await computer.send_shell_command(build_codex_session_checkpoint_command())
        if result.exit_code != 0:
            raise RuntimeError(
                "Could not archive Codex fork checkpoint: " + result.output.decode(errors="replace")
            )
        checkpoint = await computer.download(CODEX_FORK_CHECKPOINT_CONTAINER_PATH)
        checkpoint_path = bf.join(task.run_dir, CODEX_FORK_CHECKPOINT_FILENAME)
        bf.write_bytes(checkpoint_path, checkpoint)
        metadata = {
            "schema_version": 1,
            "parent_thread_id": parent_thread_id,
            "session_checkpoint": checkpoint_path,
            "source_submission": source_submission,
            "remaining_seconds": remaining_seconds,
            "auth_included": False,
            "created_at": time.time(),
        }
        bf.write_bytes(
            bf.join(task.run_dir, CODEX_FORK_METADATA_FILENAME),
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
        )
        return metadata

    @staticmethod
    def _write_exhausted_fork_metadata(
        *,
        task: PBTask,
        parent_thread_id: str | None,
        source_submission: str,
        remaining_seconds: int,
    ) -> dict[str, object]:
        """Record a deliberate no-review branch without archiving a session."""
        metadata: dict[str, object] = {
            "schema_version": 1,
            "status": "skipped-insufficient-original-budget",
            "parent_thread_id": parent_thread_id or "",
            "session_checkpoint": "",
            "source_submission": source_submission,
            "remaining_seconds": remaining_seconds,
            "auth_included": False,
            "created_at": time.time(),
        }
        bf.write_bytes(
            bf.join(task.run_dir, CODEX_FORK_METADATA_FILENAME),
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
        )
        return metadata

    async def _execute_with_checkpoints(
        self,
        *,
        computer: ComputerInterface,
        task: PBTask,
        command: str,
        start_time: float,
        event_log_path: str = CODEX_EVENT_LOG,
    ) -> ExecutionResult:
        if self.upload_interval_seconds is not None and self.upload_interval_seconds <= 0:
            raise ValueError("upload_interval_seconds must be positive or null")

        agent_task = asyncio.create_task(computer.send_shell_command(command))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {agent_task},
                    timeout=self.upload_interval_seconds,
                )
                if done:
                    return agent_task.result()

                runtime = time.time() - start_time
                try:
                    await upload_heavy_logs(
                        computer=computer,
                        agent_start_time=int(start_time),
                        agent_dir_config=AGENT_DIR_CONFIG,
                        run_dir=task.run_dir,
                        run_group_id=task.run_group_id,
                        runs_dir=task.runs_dir,
                        run_id=task.run_id,
                        runtime=runtime,
                    )
                    await upload_status(
                        start_time=int(start_time),
                        run_dir=task.run_dir,
                        status="running",
                    )
                    checkpoint_log = await self._read_event_log(
                        computer,
                        b"",
                        event_log_path,
                    )
                    self._write_run_logs(task, checkpoint_log)
                except Exception as exc:
                    logger.exception(
                        f"Could not checkpoint Codex rollout after {runtime:.1f} seconds: {exc}"
                    )
        finally:
            if not agent_task.done():
                agent_task.cancel()

    @override
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        existing_output = await check_for_existing_run(task)
        if existing_output is not None:
            return existing_output
        if self.completion_review_mode not in {"generic", "execution-log"}:
            raise ValueError(f"Unsupported completion_review_mode: {self.completion_review_mode}")
        if self.execution_feedback_timeout_seconds <= 0:
            raise ValueError("execution_feedback_timeout_seconds must be positive")
        fork_inputs = (
            self.fork_parent_thread_id,
            self.fork_session_checkpoint_path,
            self.fork_source_submission_path,
        )
        if any(value is not None for value in fork_inputs) and not all(
            value is not None for value in fork_inputs
        ):
            raise ValueError("Codex fork requires parent thread, session, and submission")
        is_fork_branch = self.fork_parent_thread_id is not None
        if is_fork_branch and not self.completion_review:
            raise ValueError("Codex fork branches require completion_review")
        if self.save_fork_checkpoint and (is_fork_branch or self.completion_review):
            raise ValueError("save_fork_checkpoint requires an unreviewed parent Agent run")

        start_time = time.time()
        await upload_status(
            start_time=int(start_time),
            run_dir=task.run_dir,
            status="running",
        )

        base_instructions = get_task_instruction_text(task)
        gpu_generation = await get_gpu_generation(computer)
        effective_instructions = build_codex_user_instructions(
            base_instructions=base_instructions,
            time_limit=self.time_limit,
            gpu_generation=gpu_generation,
        )
        await computer.upload(effective_instructions.encode(), INSTRUCTIONS_PATH)
        await sanity_check_docker(computer)

        exit_code: int | None = None
        command_output = b""
        error_msg: str | None = None
        first_fork_feedback: dict[str, object] | None = None
        first_fork_prompt: str | None = None
        first_fork_head_before: str | None = None
        command_started_at = time.time()

        if is_fork_branch:
            await self._restore_fork_checkpoint(computer)
            remaining_seconds = remaining_budget_seconds(
                time_limit_seconds=self.time_limit,
                start_time=start_time,
                now=time.time(),
            )
            if self.completion_review_mode == "execution-log":
                diagnostic_timeout = execution_feedback_timeout_seconds(
                    remaining_seconds=remaining_seconds,
                    minimum_review_seconds=self.completion_review_min_remaining_seconds,
                    maximum_execution_seconds=self.execution_feedback_timeout_seconds,
                )
                if diagnostic_timeout <= 0:
                    raise ValueError("Fork branch lacks time for execution feedback and review")
                feedback_log_path, _ = execution_feedback_paths(1)
                feedback_started_at = time.time()
                feedback_result = await computer.send_shell_command(
                    build_execution_feedback_command(
                        iteration=1,
                        timeout_seconds=diagnostic_timeout,
                    )
                )
                feedback_finished_at = time.time()
                if is_execution_feedback_infrastructure_error(feedback_result.exit_code):
                    raise RuntimeError("Execution-feedback reproduction infrastructure failed")
                first_fork_feedback = {
                    "input_version": EXECUTION_FEEDBACK_INPUT_VERSION,
                    "timeout_seconds": diagnostic_timeout,
                    "log_path": feedback_log_path,
                    "started_at": feedback_started_at,
                    "finished_at": feedback_finished_at,
                    "duration_seconds": feedback_finished_at - feedback_started_at,
                    "exit_code": feedback_result.exit_code,
                    "timed_out": feedback_result.exit_code == 124,
                    "error": None,
                }
                remaining_seconds = remaining_budget_seconds(
                    time_limit_seconds=self.time_limit,
                    start_time=start_time,
                    now=time.time(),
                )
                if remaining_seconds < self.completion_review_min_remaining_seconds:
                    raise RuntimeError("Execution feedback consumed the fork branch review budget")
                first_fork_prompt = build_execution_feedback_prompt(
                    remaining_seconds=remaining_seconds,
                    log_path=feedback_log_path,
                    reproduction_timeout_seconds=diagnostic_timeout,
                )
            else:
                first_fork_prompt = build_completion_review_prompt(remaining_seconds)
            first_fork_head_before = await get_submission_git_head(computer)
            command_started_at = time.time()
            command = build_codex_fork_command(
                parent_thread_id=self.fork_parent_thread_id or "",
                prompt=first_fork_prompt,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                time_limit=remaining_seconds,
            )
        else:
            command = build_codex_command(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                reasoning_summary=self.reasoning_summary,
                time_limit=self.time_limit,
                developer_instructions=get_system_message(
                    iterative=False,
                    code_only=task.judge.code_only,
                ),
                ephemeral=not (self.completion_review or self.save_fork_checkpoint),
            )

        try:
            result = await self._execute_with_checkpoints(
                computer=computer,
                task=task,
                command=command,
                start_time=start_time,
                event_log_path=(CODEX_REVIEW_EVENT_LOG if is_fork_branch else CODEX_EVENT_LOG),
            )
            exit_code = result.exit_code
            command_output = result.output
            if exit_code == 124:
                error_msg = f"Codex rollout timed out after {self.time_limit} seconds"
            elif exit_code != 0:
                error_msg = f"Codex rollout exited with status {exit_code}"
        except Exception as exc:
            error_msg = f"Codex rollout failed: {exc}"
            logger.exception(error_msg)

        command_finished_at = time.time()
        event_log = await self._read_event_log(
            computer,
            command_output,
            CODEX_REVIEW_EVENT_LOG if is_fork_branch else CODEX_EVENT_LOG,
        )
        if is_fork_branch and exit_code == 0 and extract_codex_thread_id(event_log) is None:
            exit_code = 1
            error_msg = "Codex fork did not emit a branch thread id"
        initial_exit_code = exit_code
        fork_checkpoint_metadata: dict[str, object] | None = None
        if self.save_fork_checkpoint:
            fork_remaining_seconds = remaining_budget_seconds(
                time_limit_seconds=self.time_limit,
                start_time=start_time,
                now=time.time(),
            )
            parent_thread_id = extract_codex_thread_id(event_log)
            should_inherit_without_review = (
                fork_remaining_seconds < self.completion_review_min_remaining_seconds
            )
            if exit_code == 0 or should_inherit_without_review:
                if parent_thread_id is None and not should_inherit_without_review:
                    exit_code = 1
                    error_msg = "Codex parent rollout did not emit a thread id"
                    source_submission = None
                else:
                    await upload_heavy_logs(
                        computer=computer,
                        agent_start_time=int(start_time),
                        agent_dir_config=AGENT_DIR_CONFIG,
                        run_dir=task.run_dir,
                        run_group_id=task.run_group_id,
                        runs_dir=task.runs_dir,
                        run_id=task.run_id,
                        runtime=time.time() - start_time,
                    )
                    source_submission = snapshot_initial_submission(task.run_dir)
                if source_submission is None:
                    if exit_code == 0:
                        exit_code = 1
                        error_msg = "Codex parent rollout did not produce a submission"
                elif should_inherit_without_review:
                    fork_checkpoint_metadata = self._write_exhausted_fork_metadata(
                        task=task,
                        parent_thread_id=parent_thread_id,
                        source_submission=source_submission,
                        remaining_seconds=fork_remaining_seconds,
                    )
                else:
                    try:
                        fork_checkpoint_metadata = await self._save_fork_checkpoint(
                            computer=computer,
                            task=task,
                            parent_thread_id=parent_thread_id,
                            source_submission=source_submission,
                            remaining_seconds=fork_remaining_seconds,
                        )
                    except Exception as exc:
                        exit_code = 1
                        error_msg = f"Could not save Codex fork checkpoint: {exc}"
                        logger.exception(error_msg)
        completion_review_metadata: dict[str, object] = {
            "enabled": self.completion_review,
            "performed": False,
            "mode": self.completion_review_mode,
            "budget_policy": "shared-agent-and-diagnostic-wall-clock",
            "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
            "execution_feedback_timeout_seconds": (self.execution_feedback_timeout_seconds),
            "initial_submission": None,
            "review_count": 0,
            "iterations": [],
            "quick_unchanged_max_seconds": QUICK_UNCHANGED_MAX_SECONDS,
            "max_consecutive_quick_unchanged": MAX_CONSECUTIVE_QUICK_UNCHANGED,
            "forked_from_id": self.fork_parent_thread_id,
        }
        if self.completion_review_mode == "execution-log":
            completion_review_metadata["execution_feedback_input_version"] = (
                EXECUTION_FEEDBACK_INPUT_VERSION
            )
        record_completion_review = exit_code == 0 or (
            is_fork_branch and exit_code == 124
        )
        if self.completion_review and record_completion_review:
            remaining_seconds = remaining_budget_seconds(
                time_limit_seconds=self.time_limit,
                start_time=start_time,
                now=time.time(),
            )
            completion_review_metadata["remaining_seconds_at_decision"] = remaining_seconds
            seed_iterations: list[dict[str, object]] = []
            seed_consecutive_quick_unchanged = 0
            if is_fork_branch:
                branch_thread_id = extract_codex_thread_id(event_log)
                first_fork_head_after = await get_submission_git_head(computer)
                first_fork_quick_unchanged = is_quick_unchanged_review(
                    head_before=first_fork_head_before,
                    head_after=first_fork_head_after,
                    started_at=command_started_at,
                    finished_at=command_finished_at,
                )
                seed_consecutive_quick_unchanged = int(first_fork_quick_unchanged)
                seed_iterations.append(
                    {
                        "index": 1,
                        "operation": "fork",
                        "prompt": first_fork_prompt,
                        "remaining_seconds_at_start": max(
                            0,
                            self.time_limit - int(command_started_at - start_time),
                        ),
                        "started_at": command_started_at,
                        "finished_at": command_finished_at,
                        "exit_code": exit_code,
                        "duration_seconds": command_finished_at - command_started_at,
                        "git_head_before": first_fork_head_before,
                        "git_head_after": first_fork_head_after,
                        "quick_unchanged": first_fork_quick_unchanged,
                        "consecutive_quick_unchanged": (seed_consecutive_quick_unchanged),
                        **(
                            {"execution_feedback": first_fork_feedback}
                            if first_fork_feedback is not None
                            else {}
                        ),
                    }
                )
                completion_review_metadata.update(
                    {
                        "performed": True,
                        "thread_id": branch_thread_id,
                        "prompt": first_fork_prompt,
                        "started_at": command_started_at,
                        "finished_at": command_finished_at,
                        "exit_code": exit_code,
                        "initial_submission": self.fork_source_submission_path,
                        "review_count": 1,
                        "iterations": seed_iterations,
                        "consecutive_quick_unchanged": (seed_consecutive_quick_unchanged),
                    }
                )
            if is_fork_branch and exit_code != 0:
                completion_review_metadata["completion_reason"] = "review-exit-nonzero"
            elif remaining_seconds < self.completion_review_min_remaining_seconds:
                if is_fork_branch:
                    completion_review_metadata["completion_reason"] = "insufficient-original-budget"
                else:
                    completion_review_metadata["skip_reason"] = "insufficient-original-budget"
            else:
                thread_id = extract_codex_thread_id(event_log)
                if thread_id is None:
                    completion_review_metadata["skip_reason"] = "missing-codex-thread-id"
                else:
                    await upload_heavy_logs(
                        computer=computer,
                        agent_start_time=int(start_time),
                        agent_dir_config=AGENT_DIR_CONFIG,
                        run_dir=task.run_dir,
                        run_group_id=task.run_group_id,
                        runs_dir=task.runs_dir,
                        run_id=task.run_id,
                        runtime=time.time() - start_time,
                    )
                    initial_submission = (
                        self.fork_source_submission_path
                        if is_fork_branch
                        else snapshot_initial_submission(task.run_dir)
                    )
                    iterations = seed_iterations
                    consecutive_quick_unchanged = seed_consecutive_quick_unchanged
                    while (
                        exit_code == 0
                        and remaining_seconds >= self.completion_review_min_remaining_seconds
                    ):
                        execution_feedback: dict[str, object] | None = None
                        if self.completion_review_mode == "execution-log":
                            diagnostic_timeout = execution_feedback_timeout_seconds(
                                remaining_seconds=remaining_seconds,
                                minimum_review_seconds=(
                                    self.completion_review_min_remaining_seconds
                                ),
                                maximum_execution_seconds=(self.execution_feedback_timeout_seconds),
                            )
                            if diagnostic_timeout <= 0:
                                completion_review_metadata["completion_reason"] = (
                                    "insufficient-budget-for-execution-feedback"
                                )
                                break
                            feedback_index = len(iterations) + 1
                            feedback_log_path, _ = execution_feedback_paths(feedback_index)
                            feedback_command = build_execution_feedback_command(
                                iteration=feedback_index,
                                timeout_seconds=diagnostic_timeout,
                            )
                            feedback_started_at = time.time()
                            try:
                                feedback_result = await computer.send_shell_command(
                                    feedback_command
                                )
                                feedback_exit_code = feedback_result.exit_code
                                feedback_error = (
                                    "Execution-feedback reproduction infrastructure failed"
                                    if is_execution_feedback_infrastructure_error(
                                        feedback_exit_code
                                    )
                                    else None
                                )
                            except Exception as exc:
                                feedback_exit_code = 1
                                feedback_error = str(exc)
                                error_msg = f"Execution feedback failed: {exc}"
                                logger.exception(error_msg)
                            feedback_finished_at = time.time()
                            execution_feedback = {
                                "input_version": EXECUTION_FEEDBACK_INPUT_VERSION,
                                "timeout_seconds": diagnostic_timeout,
                                "log_path": feedback_log_path,
                                "started_at": feedback_started_at,
                                "finished_at": feedback_finished_at,
                                "duration_seconds": (feedback_finished_at - feedback_started_at),
                                "exit_code": feedback_exit_code,
                                "timed_out": feedback_exit_code == 124,
                                "error": feedback_error,
                            }
                            if feedback_error is not None:
                                exit_code = 1
                                completion_review_metadata["completion_reason"] = (
                                    "execution-feedback-infrastructure-error"
                                )
                                break
                            remaining_seconds = remaining_budget_seconds(
                                time_limit_seconds=self.time_limit,
                                start_time=start_time,
                                now=time.time(),
                            )
                            completion_review_metadata[
                                "remaining_seconds_after_execution_feedback"
                            ] = remaining_seconds
                            if remaining_seconds < self.completion_review_min_remaining_seconds:
                                completion_review_metadata["completion_reason"] = (
                                    "insufficient-original-budget"
                                )
                                break
                            review_prompt = build_execution_feedback_prompt(
                                remaining_seconds=remaining_seconds,
                                log_path=feedback_log_path,
                                reproduction_timeout_seconds=diagnostic_timeout,
                            )
                        else:
                            review_prompt = build_completion_review_prompt(remaining_seconds)
                        review_command = build_codex_resume_command(
                            thread_id=thread_id,
                            prompt=review_prompt,
                            model=self.model,
                            reasoning_effort=self.reasoning_effort,
                            reasoning_summary=self.reasoning_summary,
                            time_limit=remaining_seconds,
                        )
                        head_before = await get_submission_git_head(computer)
                        review_started_at = time.time()
                        try:
                            review_result = await self._execute_with_checkpoints(
                                computer=computer,
                                task=task,
                                command=review_command,
                                start_time=start_time,
                                event_log_path=CODEX_REVIEW_EVENT_LOG,
                            )
                            review_log = await self._read_event_log(
                                computer,
                                review_result.output,
                                CODEX_REVIEW_EVENT_LOG,
                            )
                            event_log = event_log.rstrip() + b"\n" + review_log
                            exit_code = review_result.exit_code
                            if exit_code == 124:
                                error_msg = (
                                    "Codex completion review exhausted the remaining "
                                    "original budget"
                                )
                            elif exit_code != 0:
                                error_msg = (
                                    f"Codex completion review exited with status {exit_code}"
                                )
                            else:
                                error_msg = None
                        except Exception as exc:
                            exit_code = 1
                            error_msg = f"Codex completion review failed: {exc}"
                            logger.exception(error_msg)
                        review_finished_at = time.time()
                        head_after = await get_submission_git_head(computer)
                        quick_unchanged = is_quick_unchanged_review(
                            head_before=head_before,
                            head_after=head_after,
                            started_at=review_started_at,
                            finished_at=review_finished_at,
                        )
                        consecutive_quick_unchanged = (
                            consecutive_quick_unchanged + 1 if quick_unchanged else 0
                        )
                        iterations.append(
                            {
                                "index": len(iterations) + 1,
                                "prompt": review_prompt,
                                "remaining_seconds_at_start": remaining_seconds,
                                "started_at": review_started_at,
                                "finished_at": review_finished_at,
                                "exit_code": exit_code,
                                "duration_seconds": review_finished_at - review_started_at,
                                "git_head_before": head_before,
                                "git_head_after": head_after,
                                "quick_unchanged": quick_unchanged,
                                "consecutive_quick_unchanged": consecutive_quick_unchanged,
                                **(
                                    {"execution_feedback": execution_feedback}
                                    if execution_feedback is not None
                                    else {}
                                ),
                            }
                        )
                        completion_review_metadata.update(
                            {
                                "performed": True,
                                "thread_id": thread_id,
                                "prompt": review_prompt,
                                "remaining_seconds_at_start": iterations[0][
                                    "remaining_seconds_at_start"
                                ],
                                "started_at": iterations[0]["started_at"],
                                "finished_at": review_finished_at,
                                "exit_code": exit_code,
                                "initial_submission": initial_submission,
                                "review_count": len(iterations),
                                "iterations": iterations,
                                "consecutive_quick_unchanged": (consecutive_quick_unchanged),
                            }
                        )
                        if exit_code != 0:
                            completion_review_metadata["completion_reason"] = "review-exit-nonzero"
                            break
                        if consecutive_quick_unchanged >= MAX_CONSECUTIVE_QUICK_UNCHANGED:
                            completion_review_metadata["completion_reason"] = (
                                "consecutive-quick-unchanged"
                            )
                            break
                        remaining_seconds = remaining_budget_seconds(
                            time_limit_seconds=self.time_limit,
                            start_time=start_time,
                            now=time.time(),
                        )
                        completion_review_metadata["remaining_seconds_at_decision"] = (
                            remaining_seconds
                        )
                    if (
                        exit_code == 0
                        and remaining_seconds < self.completion_review_min_remaining_seconds
                    ):
                        completion_review_metadata["completion_reason"] = (
                            "insufficient-original-budget"
                        )

        end_time = time.time()
        if self.completion_review:
            write_completion_review_metadata(task.run_dir, completion_review_metadata)
        self._write_run_logs(task, event_log)
        try:
            await self._persist_refreshed_auth(computer)
        except Exception as exc:
            auth_error = f"Could not persist refreshed Codex authentication: {exc}"
            error_msg = f"{error_msg}; {auth_error}" if error_msg else auth_error
            logger.exception(auth_error)

        rollout_metadata = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_summary": self.reasoning_summary,
            "time_limit_seconds": self.time_limit,
            "time_start": start_time,
            "time_end": end_time,
            "runtime_in_seconds": end_time - start_time,
            "exit_code": exit_code,
            "initial_exit_code": initial_exit_code,
            "error_msg": error_msg,
            "command": command,
            "gpu_generation": gpu_generation,
            "operating_instructions": "basicagent_system_as_codex_developer",
            "url_observations_file": URL_OBSERVATIONS_FILENAME,
            "completion_review": completion_review_metadata,
            "thread_id": extract_codex_thread_id(event_log),
            "forked_from_id": self.fork_parent_thread_id,
            "fork_checkpoint": fork_checkpoint_metadata,
        }
        await computer.upload(
            json.dumps(rollout_metadata, indent=2).encode(),
            CODEX_ROLLOUT_METADATA,
        )
        try:
            await upload_heavy_logs(
                computer=computer,
                agent_start_time=int(start_time),
                agent_dir_config=AGENT_DIR_CONFIG,
                run_dir=task.run_dir,
                run_group_id=task.run_group_id,
                runs_dir=task.runs_dir,
                run_id=task.run_id,
                runtime=end_time - start_time,
            )
        except Exception as exc:
            logger.exception(f"Could not upload final Codex submission checkpoint: {exc}")
        await upload_status(
            start_time=int(start_time),
            end_time=int(end_time),
            run_dir=task.run_dir,
            status="done",
        )

        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=end_time,
            runtime_in_seconds=end_time - start_time,
            error_msg=error_msg,
            status_exists=bf.exists(bf.join(task.run_dir, "status.json")),
        )
