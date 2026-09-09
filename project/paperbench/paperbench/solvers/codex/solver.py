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
from paperbench.constants import AGENT_DIR_CONFIG, LOGS_DIR, WORKSPACE_BASE
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
    build_completion_review_prompt,
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

    async def _execute_with_checkpoints(
        self,
        *,
        computer: ComputerInterface,
        task: PBTask,
        command: str,
        start_time: float,
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
                    checkpoint_log = await self._read_event_log(computer, b"")
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

        command = build_codex_command(
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            reasoning_summary=self.reasoning_summary,
            time_limit=self.time_limit,
            developer_instructions=get_system_message(
                iterative=False,
                code_only=task.judge.code_only,
            ),
            ephemeral=not self.completion_review,
        )
        exit_code: int | None = None
        command_output = b""
        error_msg: str | None = None

        try:
            result = await self._execute_with_checkpoints(
                computer=computer,
                task=task,
                command=command,
                start_time=start_time,
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

        event_log = await self._read_event_log(computer, command_output)
        initial_exit_code = exit_code
        completion_review_metadata: dict[str, object] = {
            "enabled": self.completion_review,
            "performed": False,
            "minimum_remaining_seconds": self.completion_review_min_remaining_seconds,
            "initial_submission": None,
            "review_count": 0,
            "iterations": [],
        }
        if self.completion_review and exit_code == 0:
            remaining_seconds = remaining_budget_seconds(
                time_limit_seconds=self.time_limit,
                start_time=start_time,
                now=time.time(),
            )
            completion_review_metadata["remaining_seconds_at_decision"] = remaining_seconds
            if remaining_seconds < self.completion_review_min_remaining_seconds:
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
                    initial_submission = snapshot_initial_submission(task.run_dir)
                    iterations: list[dict[str, object]] = []
                    while (
                        exit_code == 0
                        and remaining_seconds >= self.completion_review_min_remaining_seconds
                    ):
                        review_prompt = build_completion_review_prompt(remaining_seconds)
                        review_command = build_codex_resume_command(
                            thread_id=thread_id,
                            prompt=review_prompt,
                            model=self.model,
                            reasoning_effort=self.reasoning_effort,
                            reasoning_summary=self.reasoning_summary,
                            time_limit=remaining_seconds,
                        )
                        review_started_at = time.time()
                        try:
                            review_result = await self._execute_with_checkpoints(
                                computer=computer,
                                task=task,
                                command=review_command,
                                start_time=start_time,
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
                        iterations.append(
                            {
                                "index": len(iterations) + 1,
                                "prompt": review_prompt,
                                "remaining_seconds_at_start": remaining_seconds,
                                "started_at": review_started_at,
                                "finished_at": review_finished_at,
                                "exit_code": exit_code,
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
                            }
                        )
                        if exit_code != 0:
                            completion_review_metadata["completion_reason"] = "review-exit-nonzero"
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
