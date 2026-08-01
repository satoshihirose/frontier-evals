from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Unpack, cast

import tiktoken
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessage
from openai.types.chat.completion_create_params import ResponseFormat
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from tokenizers import Tokenizer
from typing_extensions import override

from paperbench.judge.structured_output import make_strict_json_schema


def _default_tokenizer_path() -> Path | None:
    configured = os.getenv("QWEN_JUDGE_TOKENIZER_PATH")
    return Path(configured) if configured else None


class _HuggingFaceTokenEncoder:
    """Expose the tokenizers API through the subset used by SimpleJudge."""

    def __init__(self, tokenizer_path: Path) -> None:
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(
        self,
        text: str,
        *,
        disallowed_special: object = (),
    ) -> list[int]:
        del disallowed_special
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=False)


class _LeastInFlightRankRouter:
    """Assign requests to the least-used vLLM data-parallel rank."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._in_flight = [0] * size
        self._next_rank = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def rank(self) -> AsyncIterator[int | None]:
        if self.size == 1:
            yield None
            return

        async with self._lock:
            minimum = min(self._in_flight)
            chosen_rank = next(
                rank
                for offset in range(self.size)
                if self._in_flight[
                    rank := (self._next_rank + offset) % self.size
                ]
                == minimum
            )
            self._in_flight[chosen_rank] += 1
            self._next_rank = (chosen_rank + 1) % self.size

        try:
            yield chosen_rank
        finally:
            async with self._lock:
                self._in_flight[chosen_rank] -= 1


class LocalJudgeCompleter(TurnCompleter):
    """Call a local vLLM server through its OpenAI-compatible chat API."""

    encoding_name = "o200k_base"

    class Config(TurnCompleter.Config):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        backend: Literal["local_vllm"] = "local_vllm"
        model: str = "Qwen/Qwen3.6-27B"
        base_url: str = "http://127.0.0.1:58137/v1"
        api_key_env: str = "QWEN_JUDGE_API_KEY"
        response_format: type[BaseModel] | None = None
        temperature: float = Field(default=1.0, ge=0)
        top_p: float = Field(default=0.95, gt=0, le=1)
        presence_penalty: float = Field(default=1.5, ge=-2, le=2)
        max_tokens: int | None = Field(default=None, gt=0)
        seed: int | None = 0
        timeout_seconds: float = Field(default=900.0, gt=0)
        max_retries: int = Field(default=1, ge=0)
        max_concurrency: int = Field(default=4, gt=0)
        data_parallel_size: int = Field(default=1, gt=0)
        context_window: int = Field(default=200_000, gt=0)
        tokenizer_path: Path | None = Field(default_factory=_default_tokenizer_path)
        _rank_router: _LeastInFlightRankRouter = PrivateAttr()

        def model_post_init(self, __context: Any) -> None:
            del __context
            self._rank_router = _LeastInFlightRankRouter(self.data_parallel_size)

        @override
        def build(self, *, client: Any | None = None) -> LocalJudgeCompleter:
            return LocalJudgeCompleter(
                model=self.model,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                response_format=self.response_format,
                temperature=self.temperature,
                top_p=self.top_p,
                presence_penalty=self.presence_penalty,
                max_tokens=self.max_tokens,
                seed=self.seed,
                timeout_seconds=self.timeout_seconds,
                max_retries=self.max_retries,
                max_concurrency=self.max_concurrency,
                rank_router=self._rank_router,
                context_window=self.context_window,
                tokenizer_path=self.tokenizer_path,
                client=client,
            )

        def with_response_format(
            self, response_format: type[BaseModel]
        ) -> LocalJudgeCompleter.Config:
            return self.model_copy(update={"response_format": response_format})

        def checkpoint_identity(self) -> dict[str, Any]:
            return {
                "backend": self.backend,
                "model": self.model,
                "response_format": (
                    f"{self.response_format.__module__}:{self.response_format.__qualname__}"
                    if self.response_format is not None
                    else None
                ),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "presence_penalty": self.presence_penalty,
                "max_tokens": self.max_tokens,
                "seed": self.seed,
                "context_window": self.context_window,
                "tokenizer": self._tokenizer_identity(),
            }

        def _tokenizer_identity(self) -> dict[str, str]:
            if self.tokenizer_path is None:
                return {"type": "tiktoken", "name": LocalJudgeCompleter.encoding_name}
            return {
                "type": "huggingface_tokenizer_json",
                "sha256": hashlib.sha256(self.tokenizer_path.read_bytes()).hexdigest(),
            }

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        response_format: type[BaseModel] | None,
        temperature: float,
        top_p: float,
        presence_penalty: float,
        max_tokens: int | None,
        seed: int | None,
        timeout_seconds: float,
        max_retries: int,
        max_concurrency: int,
        rank_router: _LeastInFlightRankRouter,
        context_window: int,
        tokenizer_path: Path | None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/") + "/"
        self.api_key_env = api_key_env
        self.response_format = response_format
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.max_tokens = max_tokens
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_concurrency = max_concurrency
        self._rank_router = rank_router
        self.n_ctx = context_window
        self._token_encoder = (
            _HuggingFaceTokenEncoder(tokenizer_path)
            if tokenizer_path is not None
            else tiktoken.get_encoding(self.encoding_name)
        )
        self._client = client

    def get_token_encoder(self) -> Any:
        return self._token_encoder

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.getenv(self.api_key_env, "EMPTY"),
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                max_retries=self.max_retries,
            )
        return self._client

    def _response_format_param(self) -> ResponseFormat | None:
        if self.response_format is None:
            return None
        schema = make_strict_json_schema(self.response_format.model_json_schema())
        return cast(
            ResponseFormat,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_format.__name__,
                    "schema": schema,
                    "strict": True,
                },
            },
        )

    @override
    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> LocalJudgeCompleter.Completion:
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
    ) -> LocalJudgeCompleter.Completion:
        del params
        request: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "seed": self.seed,
        }
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens
        response_format = self._response_format_param()
        if response_format is not None:
            request["response_format"] = response_format

        async with self._rank_router.rank() as rank:
            if rank is not None:
                request["extra_headers"] = {"X-data-parallel-rank": str(rank)}
            response = await self._get_client().chat.completions.create(**request)
        if not response.choices:
            raise RuntimeError("Local vLLM judge returned no completion choices")
        message = response.choices[0].message
        content = message.content
        if not content:
            raise RuntimeError("Local vLLM judge returned an empty final response")
        if self.response_format is not None:
            parsed = self.response_format.model_validate_json(content)
            content = parsed.model_dump_json()
            message = ChatCompletionMessage(role="assistant", content=content)

        return self.Completion(
            input_conversation=conversation,
            output_messages=[message],
            usage=response.usage,
        )
