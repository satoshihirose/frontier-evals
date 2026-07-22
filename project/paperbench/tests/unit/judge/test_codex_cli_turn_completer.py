from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)

from paperbench.grade import get_completer_backend
from paperbench.judge.codex_cli_turn_completer import (
    CodexCliTurnCompleter,
    build_codex_cli_args,
)
from paperbench.judge.simple import ParsedJudgeResponseInt, SimpleJudge
from paperbench.nano.eval import requires_grader_openai_api_key
from paperbench.rubric.tasks import TaskNode


def test_codex_cli_default_model_is_repeatable_judge_model() -> None:
    config = CodexCliTurnCompleter.Config()

    assert config.model == "gpt-5.4"
    assert config.reasoning_effort == "high"
    assert config.context_window == 272_000


def test_codex_cli_args_use_subscription_auth_and_structured_output(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "schema.json"
    output_path = tmp_path / "output.json"

    args = build_codex_cli_args(
        codex_path="codex",
        model="gpt-5.4",
        reasoning_effort="high",
        work_dir=tmp_path,
        output_path=output_path,
        schema_path=schema_path,
    )

    assert args[:2] == ["codex", "exec"]
    assert "--ephemeral" in args
    assert "--ignore-user-config" in args
    assert "--sandbox" in args
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("--output-schema") + 1] == str(schema_path)
    assert args[args.index("--output-last-message") + 1] == str(output_path)
    assert args[-1] == "-"


@pytest.mark.asyncio
async def test_codex_cli_completion_removes_api_credentials_and_returns_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    codex_home = tmp_path / ".codex"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text("{}", encoding="utf-8")
    observed: dict[str, Any] = {}

    calls: list[dict[str, Any]] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, args: list[str], env: dict[str, str]) -> None:
            self.args = args
            self.env = env

        async def communicate(self, prompt: bytes | None = None) -> tuple[bytes, bytes]:
            if self.args[1:3] == ["login", "status"]:
                return b"Logged in using ChatGPT\n", b""
            assert prompt is not None
            observed["prompt"] = prompt.decode()
            observed["args"] = self.args
            observed["env"] = self.env
            runtime_auth_path = Path(self.env["CODEX_HOME"]) / "auth.json"
            observed["auth_content"] = runtime_auth_path.read_text(encoding="utf-8")
            observed["auth_mode"] = runtime_auth_path.stat().st_mode & 0o777
            observed["schema"] = json.loads(
                Path(self.args[self.args.index("--output-schema") + 1]).read_text(encoding="utf-8")
            )
            output_path = Path(self.args[self.args.index("--output-last-message") + 1])
            output_path.write_text(
                '{"valid_score":true,"score":1,"explanation":"Satisfied."}',
                encoding="utf-8",
            )
            return b"", b""

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        call = {"args": list(args), "env": kwargs["env"]}
        calls.append(call)
        return FakeProcess(call["args"], call["env"])

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("GRADER_OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setattr(
        "paperbench.judge.codex_cli_turn_completer.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    completer = CodexCliTurnCompleter.Config(
        model="gpt-5.4",
        codex_home=str(codex_home),
        response_format=ParsedJudgeResponseInt,
    ).build()

    completion = await completer.async_completion(
        [
            {"role": "system", "content": "Grade the submission."},
            {"role": "user", "content": "Return the score."},
        ]
    )

    assert "OPENAI_API_KEY" not in observed["env"]
    assert "GRADER_OPENAI_API_KEY" not in observed["env"]
    assert observed["env"]["CODEX_HOME"] != str(codex_home)
    assert observed["auth_content"] == "{}"
    assert observed["auth_mode"] == 0o600
    assert json.loads(completion.output_messages[0].content or "{}")["score"] == 1
    assert observed["prompt"].startswith(
        "Respond to the following conversation as the assistant. Do not use tools.\n\n"
    )
    assert "Grade the submission." in observed["prompt"]
    assert "Do not inspect files" not in observed["prompt"]
    assert observed["schema"]["additionalProperties"] is False
    assert [call["args"][1:3] for call in calls] == [
        ["login", "status"],
        ["exec", "--model"],
    ]


def test_simple_judge_uses_codex_cli_for_main_and_parser_calls(
    tmp_path: Path,
) -> None:
    (tmp_path / "paper.md").write_text("", encoding="utf-8")
    config = CodexCliTurnCompleter.Config(
        model="gpt-5.4",
        codex_home=str(tmp_path),
        max_concurrency=3,
    )
    judge = SimpleJudge(
        paper_path=tmp_path / "paper.pdf",
        rubric=TaskNode(
            id="root",
            requirements="Root",
            weight=1,
            sub_tasks=[],
            task_category="Code Development",
        ),
        addendum=None,
        judge_addendum=None,
        submission_dir=tmp_path,
        paper_md=tmp_path / "paper.md",
        completer_config=config,
    )

    assert isinstance(judge.completer, CodexCliTurnCompleter)
    assert isinstance(judge.int_completer, CodexCliTurnCompleter)
    assert isinstance(judge.float_completer, CodexCliTurnCompleter)
    assert judge.int_completer.response_format is ParsedJudgeResponseInt
    assert judge.leaf_semaphore._value == 3


def test_only_api_backend_requires_grader_openai_api_key() -> None:
    assert requires_grader_openai_api_key(CodexCliTurnCompleter.Config(model="gpt-5.4")) is False
    assert (
        requires_grader_openai_api_key(OpenAICompletionsTurnCompleter.Config(model="gpt-4o"))
        is True
    )
    assert get_completer_backend(CodexCliTurnCompleter.Config(model="gpt-5.4")) == (
        "paperbench.judge.codex_cli_turn_completer:CodexCliTurnCompleter.Config"
    )
