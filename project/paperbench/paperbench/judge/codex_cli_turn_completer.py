from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, Unpack

import structlog.stdlib
import tiktoken
from openai.types.chat import ChatCompletionMessage
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from paperbench.judge.structured_output import make_strict_json_schema

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]
logger = structlog.stdlib.get_logger(component=__name__)


async def terminate_process_group(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 5,
) -> None:
    """Terminate a Codex process and every descendant in its process group."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await process.wait()


def build_codex_cli_args(
    *,
    codex_path: str,
    model: str,
    reasoning_effort: ReasoningEffort,
    work_dir: Path,
    output_path: Path,
    schema_path: Path | None,
) -> list[str]:
    args = [
        codex_path,
        "exec",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-c",
        "features.shell_tool=false",
        "-c",
        'web_search="disabled"',
        "-c",
        "agents.enabled=false",
        "-c",
        "features.multi_agent=false",
        "-c",
        "features.apps=false",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--cd",
        str(work_dir),
        "--output-last-message",
        str(output_path),
    ]
    if schema_path is not None:
        args.extend(["--output-schema", str(schema_path)])
    args.append("-")
    return args


def _render_conversation(conversation: TurnCompleter.RuntimeConversation) -> str:
    return (
        "Respond to the following conversation as the assistant.\n\n"
        + json.dumps(conversation, ensure_ascii=False, default=str)
    )


class CodexCliTurnCompleter(TurnCompleter):
    """Use a ChatGPT-authenticated Codex CLI process as a PaperBench completer."""

    encoding_name = "o200k_base"

    class Config(TurnCompleter.Config):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        backend: Literal["codex_cli"] = "codex_cli"
        model: str = "gpt-5.4"
        reasoning_effort: ReasoningEffort = "high"
        codex_path: str = "codex"
        codex_home: str = "~/.codex"
        response_format: type[BaseModel] | None = None
        timeout_seconds: float = Field(default=900.0, gt=0)
        timeout_retries: int = Field(default=1, ge=0)
        max_concurrency: int = Field(default=4, gt=0)
        context_window: int = Field(default=272_000, gt=0)

        @override
        def build(self) -> CodexCliTurnCompleter:
            return CodexCliTurnCompleter(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                codex_path=self.codex_path,
                codex_home=Path(self.codex_home).expanduser(),
                response_format=self.response_format,
                timeout_seconds=self.timeout_seconds,
                timeout_retries=self.timeout_retries,
                max_concurrency=self.max_concurrency,
                context_window=self.context_window,
            )

        def with_response_format(
            self, response_format: type[BaseModel]
        ) -> CodexCliTurnCompleter.Config:
            return self.model_copy(update={"response_format": response_format})

        def checkpoint_identity(self) -> dict[str, Any]:
            return {
                "backend": self.backend,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "response_format": (
                    f"{self.response_format.__module__}:{self.response_format.__qualname__}"
                    if self.response_format is not None
                    else None
                ),
                "context_window": self.context_window,
            }

    class Completion(TurnCompleter.Completion):
        pass

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: ReasoningEffort,
        codex_path: str,
        codex_home: Path,
        response_format: type[BaseModel] | None,
        timeout_seconds: float,
        timeout_retries: int,
        max_concurrency: int,
        context_window: int,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.codex_path = codex_path
        self.codex_home = codex_home.resolve()
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds
        self.timeout_retries = timeout_retries
        self.max_concurrency = max_concurrency
        self.n_ctx = context_window
        self._encoder = tiktoken.get_encoding(self.encoding_name)
        self._subscription_auth_verified = False
        self._subscription_auth_lock = asyncio.Lock()

    def _subscription_env(self, runtime_codex_home: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("GRADER_OPENAI_API_KEY", None)
        env["CODEX_HOME"] = str(runtime_codex_home)
        return env

    async def _verify_subscription_auth(self, runtime_codex_home: Path) -> None:
        if self._subscription_auth_verified:
            return
        async with self._subscription_auth_lock:
            if self._subscription_auth_verified:
                return
            process = await asyncio.create_subprocess_exec(
                self.codex_path,
                "login",
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subscription_env(runtime_codex_home),
            )
            stdout, stderr = await process.communicate()
            status = (stdout + stderr).decode(errors="replace")
            if process.returncode != 0 or "chatgpt" not in status.lower():
                raise RuntimeError(
                    "Codex CLI judge requires ChatGPT subscription authentication; "
                    "run `codex login` and select Sign in with ChatGPT"
                )
            self._subscription_auth_verified = True

    @override
    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> CodexCliTurnCompleter.Completion:
        del params
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_completion(conversation))
        raise RuntimeError("completion() cannot run inside an active event loop")

    @override
    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> CodexCliTurnCompleter.Completion:
        del params
        for attempt in range(self.timeout_retries + 1):
            try:
                return await self._async_completion_once(conversation)
            except TimeoutError:
                if attempt >= self.timeout_retries:
                    raise
                logger.warning(
                    "Retrying timed-out Codex CLI judge call",
                    attempt=attempt + 1,
                    max_attempts=self.timeout_retries + 1,
                )
        raise AssertionError("unreachable")

    async def _async_completion_once(
        self,
        conversation: TurnCompleter.RuntimeConversation,
    ) -> CodexCliTurnCompleter.Completion:
        auth_path = self.codex_home / "auth.json"
        if not auth_path.is_file():
            raise FileNotFoundError(f"Codex ChatGPT authentication file not found: {auth_path}")

        with tempfile.TemporaryDirectory(prefix="paperbench-codex-judge-") as tmp:
            work_dir = Path(tmp)
            runtime_codex_home = work_dir / "codex-home"
            runtime_codex_home.mkdir()
            runtime_auth_path = runtime_codex_home / "auth.json"
            shutil.copyfile(auth_path, runtime_auth_path)
            runtime_auth_path.chmod(0o600)
            await self._verify_subscription_auth(runtime_codex_home)
            output_path = work_dir / "last-message.txt"
            schema_path: Path | None = None
            if self.response_format is not None:
                schema_path = work_dir / "output-schema.json"
                schema_path.write_text(
                    json.dumps(make_strict_json_schema(self.response_format.model_json_schema())),
                    encoding="utf-8",
                )
            args = build_codex_cli_args(
                codex_path=self.codex_path,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                work_dir=work_dir,
                output_path=output_path,
                schema_path=schema_path,
            )
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._subscription_env(runtime_codex_home),
                start_new_session=True,
            )
            communication = asyncio.create_task(
                process.communicate(_render_conversation(conversation).encode())
            )
            try:
                done, _ = await asyncio.wait({communication}, timeout=self.timeout_seconds)
                if not done:
                    await terminate_process_group(process)
                    try:
                        await asyncio.wait_for(communication, timeout=5)
                    except TimeoutError:
                        communication.cancel()
                        with suppress(asyncio.CancelledError):
                            await communication
                    raise TimeoutError(
                        f"Codex CLI judge timed out after {self.timeout_seconds} seconds"
                    )
                _, stderr = communication.result()
            except asyncio.CancelledError:
                await terminate_process_group(process)
                communication.cancel()
                with suppress(asyncio.CancelledError):
                    await communication
                raise
            except TimeoutError:
                raise TimeoutError(
                    f"Codex CLI judge timed out after {self.timeout_seconds} seconds"
                ) from None
            if process.returncode != 0:
                detail = stderr.decode(errors="replace")[-4000:]
                raise RuntimeError(
                    f"Codex CLI judge exited with code {process.returncode}: {detail}"
                )
            if not output_path.is_file():
                raise RuntimeError("Codex CLI judge did not write its final response")
            content = output_path.read_text(encoding="utf-8").strip()
            if self.response_format is not None:
                parsed = self.response_format.model_validate_json(content)
                content = parsed.model_dump_json()
            return self.Completion(
                input_conversation=conversation,
                output_messages=[ChatCompletionMessage(role="assistant", content=content)],
            )
