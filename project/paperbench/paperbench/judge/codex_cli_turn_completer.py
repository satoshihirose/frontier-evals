from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal, Unpack

import tiktoken
from openai.types.chat import ChatCompletionMessage
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


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
        "Respond to the following conversation as the assistant. Do not use tools.\n\n"
        + json.dumps(conversation, ensure_ascii=False, default=str)
    )


def make_strict_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        strict = {key: make_strict_json_schema(item) for key, item in value.items()}
        if strict.get("type") == "object":
            strict["additionalProperties"] = False
        return strict
    if isinstance(value, list):
        return [make_strict_json_schema(item) for item in value]
    return value


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
                max_concurrency=self.max_concurrency,
                context_window=self.context_window,
            )

        def with_response_format(
            self, response_format: type[BaseModel]
        ) -> CodexCliTurnCompleter.Config:
            return self.model_copy(update={"response_format": response_format})

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
        max_concurrency: int,
        context_window: int,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.codex_path = codex_path
        self.codex_home = codex_home.resolve()
        self.response_format = response_format
        self.timeout_seconds = timeout_seconds
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
            )
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(_render_conversation(conversation).encode()),
                    timeout=self.timeout_seconds,
                )
            except TimeoutError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
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
