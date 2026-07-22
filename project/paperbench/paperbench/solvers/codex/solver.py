from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import blobfile as bf
import structlog.stdlib
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerRuntimeNoJupyter
from typing_extensions import override

import chz
from alcatraz.clusters.local import VolumesConfig
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ExecutionResult,
)
from paperbench.constants import AGENT_DIR_CONFIG, LOGS_DIR, WORKSPACE_BASE
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver
from paperbench.solvers.basicagent.prompts.templates import (
    additional_notes_template,
    get_system_message,
    gpu_template,
    no_gpu_template,
    time_limit_template,
)
from paperbench.solvers.basicagent.utils import get_gpu_generation
from paperbench.solvers.upload import upload_heavy_logs, upload_status
from paperbench.solvers.utils import check_for_existing_run, sanity_check_docker

logger = structlog.stdlib.get_logger(component=__name__)

CODEX_EVENT_LOG = f"{LOGS_DIR}/codex-events.jsonl"
CODEX_ROLLOUT_METADATA = f"{LOGS_DIR}/codex-rollout.json"
URL_OBSERVATIONS_FILENAME = "url-observations.jsonl"
CODEX_HOME = f"{WORKSPACE_BASE}/.codex"
CODEX_AUTH_PATH = f"{CODEX_HOME}/auth.json"
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
    type_of_processor_str = (
        gpu_template.format(type_of_processor=gpu_generation) if gpu_generation else no_gpu_template
    )
    time_limit_str = time_limit_template.format(max_time_in_hours=round(time_limit / 3600, 3))
    runtime_context = additional_notes_template.format(
        type_of_processor_str=type_of_processor_str,
        time_limit_str=time_limit_str,
        workspace_base=WORKSPACE_BASE,
    )
    return f"{base_instructions.rstrip()}\n{runtime_context}"


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
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-",
    ]
    return (
        f"{shlex.join(args)} < {shlex.quote(INSTRUCTIONS_PATH)} "
        f"> {shlex.quote(CODEX_EVENT_LOG)} 2>&1"
    )


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
        doc="Host ChatGPT auth file mounted read-only into the Codex agent container",
    )
    time_limit: int = chz.field(default=24 * 60 * 60, doc="Rollout time limit in seconds")
    upload_interval_seconds: float | None = chz.field(
        default=1800,
        doc="Seconds between submission checkpoints; set to null to disable",
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

        volumes_config = VolumesConfig()
        volumes_config["codexauth"] = {
            "bind_source": str(auth_file),
            "bind_dest": CODEX_AUTH_PATH,
            "mode": "ro",
        }
        task.volumes_config = {**(task.volumes_config or {}), **volumes_config}
        return task

    @override
    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        del task
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
                "container; the mounted auth file must come from `codex login` with ChatGPT"
            )

    async def _read_event_log(self, computer: ComputerInterface, fallback: bytes) -> bytes:
        try:
            return await computer.download(CODEX_EVENT_LOG)
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

        base_instructions = task.prompt[0].get("content")
        if not isinstance(base_instructions, str):
            raise TypeError(
                f"Expected task instructions to be str, got {type(base_instructions)!r}"
            )
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

        end_time = time.time()
        event_log = await self._read_event_log(computer, command_output)
        self._write_run_logs(task, event_log)

        rollout_metadata = {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_summary": self.reasoning_summary,
            "time_limit_seconds": self.time_limit,
            "time_start": start_time,
            "time_end": end_time,
            "runtime_in_seconds": end_time - start_time,
            "exit_code": exit_code,
            "error_msg": error_msg,
            "command": command,
            "gpu_generation": gpu_generation,
            "operating_instructions": "basicagent_system_as_codex_developer",
            "url_observations_file": URL_OBSERVATIONS_FILENAME,
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
