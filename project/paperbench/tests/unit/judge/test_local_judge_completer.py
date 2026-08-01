from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from paperbench.judge.local_judge_completer import (
    LocalJudgeCompleter,
    _HuggingFaceTokenEncoder,
)
from paperbench.judge.simple import ParsedJudgeResponseInt, SimpleJudge
from paperbench.nano.eval import requires_grader_openai_api_key
from paperbench.rubric.tasks import TaskNode


class StructuredAnswer(BaseModel):
    score: int
    explanation: str


class FakeChatCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: object) -> None:
        self.chat = SimpleNamespace(completions=FakeChatCompletions(response))


class ControlledChatCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []
        self.release = asyncio.Event()

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        await self.release.wait()
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class ControlledClient:
    def __init__(self, responses: list[object]) -> None:
        self.completions = ControlledChatCompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)


async def wait_for_calls(client: ControlledClient, count: int) -> None:
    for _ in range(100):
        if len(client.completions.calls) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"Expected {count} calls, got {len(client.completions.calls)}")


def make_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=ChatCompletionMessage(role="assistant", content=content)
            )
        ],
        usage=CompletionUsage(
            completion_tokens=7,
            prompt_tokens=11,
            total_tokens=18,
        ),
    )


def test_local_judge_defaults_target_qwen_vllm() -> None:
    config = LocalJudgeCompleter.Config()

    assert config.backend == "local_vllm"
    assert config.model == "Qwen/Qwen3.6-27B"
    assert config.base_url == "http://127.0.0.1:58137/v1"
    assert config.context_window == 200_000
    assert config.max_tokens is None
    assert config.max_concurrency == 4
    assert config.data_parallel_size == 1
    assert config.api_key_env == "QWEN_JUDGE_API_KEY"
    assert config.tokenizer_path is None


def test_local_judge_reads_tokenizer_path_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    monkeypatch.setenv("QWEN_JUDGE_TOKENIZER_PATH", str(tokenizer_path))

    assert LocalJudgeCompleter.Config().tokenizer_path == tokenizer_path


def test_checkpoint_identity_ignores_transport_and_concurrency() -> None:
    first = LocalJudgeCompleter.Config(
        base_url="http://127.0.0.1:58137/v1",
        api_key_env="QWEN_JUDGE_API_KEY",
        timeout_seconds=900,
        max_retries=1,
        max_concurrency=4,
    )
    resumed = LocalJudgeCompleter.Config(
        base_url="http://10.0.0.3:9000/v1",
        api_key_env="OTHER_LOCAL_KEY",
        timeout_seconds=60,
        max_retries=3,
        max_concurrency=1,
        data_parallel_size=2,
    )

    assert first.checkpoint_identity() == resumed.checkpoint_identity()


def test_checkpoint_identity_changes_with_generation_semantics() -> None:
    first = LocalJudgeCompleter.Config(seed=0, temperature=1.0)
    changed = LocalJudgeCompleter.Config(seed=1, temperature=1.0)

    assert first.checkpoint_identity() != changed.checkpoint_identity()


def test_checkpoint_identity_changes_with_tokenizer_content(tmp_path: Path) -> None:
    first_path = tmp_path / "first-tokenizer.json"
    same_path = tmp_path / "same-tokenizer.json"
    changed_path = tmp_path / "changed-tokenizer.json"
    first_path.write_text('{"version":"first"}', encoding="utf-8")
    same_path.write_text('{"version":"first"}', encoding="utf-8")
    changed_path.write_text('{"version":"changed"}', encoding="utf-8")

    first = LocalJudgeCompleter.Config(tokenizer_path=first_path)
    same = LocalJudgeCompleter.Config(tokenizer_path=same_path)
    changed = LocalJudgeCompleter.Config(tokenizer_path=changed_path)

    assert first.checkpoint_identity() == same.checkpoint_identity()
    assert first.checkpoint_identity() != changed.checkpoint_identity()


def write_test_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={"[UNK]": 0, "hello": 1, "world": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def test_huggingface_encoder_accepts_simple_judge_interface(tmp_path: Path) -> None:
    tokenizer_path = tmp_path / "tokenizer.json"
    write_test_tokenizer(tokenizer_path)
    encoder = _HuggingFaceTokenEncoder(tokenizer_path)

    assert encoder.encode("hello world", disallowed_special=()) == [1, 2]
    assert encoder.decode([1, 2]) == "hello world"


def test_local_client_uses_configured_endpoint_and_environment_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    sentinel = object()

    def fake_async_openai(**kwargs: Any) -> object:
        observed.update(kwargs)
        return sentinel

    monkeypatch.setenv("PRIVATE_LOCAL_JUDGE_KEY", "local-secret")
    monkeypatch.setattr(
        "paperbench.judge.local_judge_completer.AsyncOpenAI",
        fake_async_openai,
    )
    completer = LocalJudgeCompleter.Config(
        base_url="http://localhost:9000/v1",
        api_key_env="PRIVATE_LOCAL_JUDGE_KEY",
        timeout_seconds=123,
        max_retries=2,
    ).build()

    assert completer._get_client() is sentinel
    assert observed == {
        "api_key": "local-secret",
        "base_url": "http://localhost:9000/v1/",
        "timeout": 123,
        "max_retries": 2,
    }


@pytest.mark.asyncio
async def test_local_completion_preserves_messages_and_usage() -> None:
    response = make_response("SCORE: 1\nEXPLANATION: Met.")
    client = FakeClient(response)
    completer = LocalJudgeCompleter.Config().build(client=client)
    conversation: TurnCompleter.RuntimeConversation = [
        {"role": "system", "content": "Grade the submission."},
        {"role": "user", "content": "Return the score."},
    ]

    completion = await completer.async_completion(conversation)

    assert completion.output_messages[0].content == "SCORE: 1\nEXPLANATION: Met."
    assert completion.usage == response.usage
    call = client.chat.completions.calls[0]
    assert call["messages"] == conversation
    assert call["model"] == "Qwen/Qwen3.6-27B"
    assert call["seed"] == 0
    assert "max_tokens" not in call
    assert "response_format" not in call
    assert "extra_headers" not in call


@pytest.mark.asyncio
async def test_dp_rank_router_balances_concurrent_requests() -> None:
    response = make_response("SCORE: 1\nEXPLANATION: Met.")
    client = ControlledClient([response, response, response, response])
    completer = LocalJudgeCompleter.Config(data_parallel_size=2).build(client=client)
    conversation: TurnCompleter.RuntimeConversation = [
        {"role": "user", "content": "Return the score."}
    ]

    tasks = [
        asyncio.create_task(completer.async_completion(conversation))
        for _ in range(4)
    ]
    await wait_for_calls(client, 4)

    assert [
        call["extra_headers"]["X-data-parallel-rank"]
        for call in client.completions.calls
    ] == ["0", "1", "0", "1"]

    client.completions.release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_dp_rank_router_is_shared_with_structured_completers() -> None:
    main_client = ControlledClient([make_response("SCORE: 1\nEXPLANATION: Met.")])
    structured_client = ControlledClient(
        [make_response('{"score":1,"explanation":"Met."}')]
    )
    config = LocalJudgeCompleter.Config(data_parallel_size=2)
    main = config.build(client=main_client)
    structured = config.with_response_format(StructuredAnswer).build(
        client=structured_client
    )
    conversation: TurnCompleter.RuntimeConversation = [
        {"role": "user", "content": "Return the score."}
    ]

    main_task = asyncio.create_task(main.async_completion(conversation))
    await wait_for_calls(main_client, 1)
    structured_task = asyncio.create_task(structured.async_completion(conversation))
    await wait_for_calls(structured_client, 1)

    assert (
        main_client.completions.calls[0]["extra_headers"]["X-data-parallel-rank"]
        == "0"
    )
    assert (
        structured_client.completions.calls[0]["extra_headers"][
            "X-data-parallel-rank"
        ]
        == "1"
    )

    main_client.completions.release.set()
    structured_client.completions.release.set()
    await asyncio.gather(main_task, structured_task)


@pytest.mark.asyncio
async def test_dp_rank_router_releases_rank_after_failure() -> None:
    response = make_response("SCORE: 1\nEXPLANATION: Met.")
    client = ControlledClient([RuntimeError("request failed"), response, response])
    completer = LocalJudgeCompleter.Config(data_parallel_size=2).build(client=client)
    conversation: TurnCompleter.RuntimeConversation = [
        {"role": "user", "content": "Return the score."}
    ]

    first = asyncio.create_task(completer.async_completion(conversation))
    await wait_for_calls(client, 1)
    client.completions.release.set()
    with pytest.raises(RuntimeError, match="request failed"):
        await first

    await completer.async_completion(conversation)
    await completer.async_completion(conversation)

    assert [
        call["extra_headers"]["X-data-parallel-rank"]
        for call in client.completions.calls
    ] == ["0", "1", "0"]


@pytest.mark.asyncio
async def test_local_completion_sends_explicit_max_tokens_override() -> None:
    client = FakeClient(make_response("SCORE: 1\nEXPLANATION: Met."))
    completer = LocalJudgeCompleter.Config(max_tokens=2_048).build(client=client)

    await completer.async_completion(
        [{"role": "user", "content": "Return the score."}]
    )

    assert client.chat.completions.calls[0]["max_tokens"] == 2_048


@pytest.mark.asyncio
async def test_local_structured_completion_uses_json_schema_and_validates() -> None:
    client = FakeClient(make_response('{"score":1,"explanation":"Met."}'))
    config = LocalJudgeCompleter.Config().with_response_format(StructuredAnswer)
    completer = config.build(client=client)

    completion = await completer.async_completion(
        [{"role": "user", "content": "Parse this response."}]
    )

    call = client.chat.completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert (
        call["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )
    assert StructuredAnswer.model_validate_json(
        completion.output_messages[0].content or ""
    ).score == 1


@pytest.mark.asyncio
async def test_local_structured_completion_rejects_invalid_output() -> None:
    client = FakeClient(make_response('{"score":"not-an-int","explanation":"No."}'))
    completer = LocalJudgeCompleter.Config(
        response_format=StructuredAnswer
    ).build(client=client)

    with pytest.raises(ValueError):
        await completer.async_completion(
            [{"role": "user", "content": "Parse this response."}]
        )


def test_local_judge_does_not_require_official_openai_api_key() -> None:
    assert requires_grader_openai_api_key(LocalJudgeCompleter.Config()) is False


def test_simple_judge_uses_local_backend_for_main_and_parser_calls(
    tmp_path: Path,
) -> None:
    paper_md = tmp_path / "paper.md"
    paper_md.write_text("", encoding="utf-8")
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
        paper_md=paper_md,
        completer_config=LocalJudgeCompleter.Config(max_concurrency=3),
    )

    assert isinstance(judge.completer, LocalJudgeCompleter)
    assert isinstance(judge.int_completer, LocalJudgeCompleter)
    assert isinstance(judge.float_completer, LocalJudgeCompleter)
    assert judge.int_completer.response_format is ParsedJudgeResponseInt
    assert judge.leaf_semaphore._value == 3


def test_simple_judge_uses_local_model_tokenizer(tmp_path: Path) -> None:
    paper_md = tmp_path / "paper.md"
    paper_md.write_text("", encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    write_test_tokenizer(tokenizer_path)
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
        paper_md=paper_md,
        completer_config=LocalJudgeCompleter.Config(
            tokenizer_path=tokenizer_path,
        ),
    )

    assert isinstance(judge.token_encoder, _HuggingFaceTokenEncoder)
    assert judge.token_encoder.encode("hello world", disallowed_special=()) == [1, 2]


def test_simple_judge_records_local_vllm_usage(tmp_path: Path) -> None:
    paper_md = tmp_path / "paper.md"
    paper_md.write_text("", encoding="utf-8")
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
        paper_md=paper_md,
        completer_config=LocalJudgeCompleter.Config(),
    )

    usage = judge._handle_usage(
        judge.completer,
        None,
        CompletionUsage(completion_tokens=7, prompt_tokens=11, total_tokens=18),
    )

    assert usage is not None
    assert usage.to_dict() == {"Qwen/Qwen3.6-27B": {"in": 11, "out": 7}}
