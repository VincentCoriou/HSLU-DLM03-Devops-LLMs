"""This file provides abstractions for interacting with various LLM backends."""

import dataclasses
import os
from typing import Any

import openai

from hslu.dlm03.common import chat as chat_lib
from hslu.dlm03.common import types
from hslu.dlm03.util import ratelimit

_GOOGLE_OPENAI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GOOGLE_API_KEY_ENV_VAR = "GOOGLE_API_KEY"

_OPENAI_API_BASE_URL = "https://api.openai.com/v1"
_OPENAI_API_KEY_ENV_VAR = "OPENAI_API_KEY"

_LLAMA_CPP_API_KEY_ENV_VAR = "LLAMA_CPP_API_KEY"


class LLMBackend:
    """A class that provides an interface to an LLM backend.

    It uses an `openai.Client` to interact with the LLM and enforces a
    rate limit on the number of calls per minute.
    """
    _client: openai.Client
    _model: str
    _ratelimiter: ratelimit.RateLimiter

    def __init__(self, *, client: openai.Client, model: str, ratelimiter: ratelimit.RateLimiter) -> None:
        """Initializes an `LLMBackend` instance.

        Args:
            client: The OpenAI client.
            model: The model name.
            ratelimiter: The rate limiter.
        """
        self._client = client
        self._model = model
        self._ratelimiter = ratelimiter

    def __call__(self, *, response_format: type | None = None, **kwargs: Any) -> types.ModelResponse:
        """Calls the LLM backend and returns its output.

        Args:
            response_format: The type of the expected response from the model.
            **kwargs: Additional keyword arguments to pass to the API call.

        Returns:
            A `types.ModelResponse` object representing the model's response.
        """
        with self._ratelimiter:
            if response_format is not None:
                kwargs["response_format"] = response_format
                fn = self._client.chat.completions.parse
            else:
                fn = self._client.chat.completions.create
            return fn(model=self._model, **kwargs)

    def generate(self, chat: chat_lib.Chat, /, **kwargs: Any) -> types.ModelResponse:
        """Generates a response from the LLM based on the provided chat history.

        Args:
            chat: A `chat_lib.Chat` object containing the conversation history.
            **kwargs: Additional keyword arguments to pass to the `__call__` method.

        Returns:
            A `types.ModelResponse` object representing the model's response.

        """
        return self(messages=chat.messages, **kwargs)


class AsyncLLMBackend:
    """A class that provides an asynchronous interface to an LLM backend.

    It uses an `openai.AsyncClient` to interact with the LLM and enforces a
    rate limit on the number of calls per minute.
    """
    _client: openai.AsyncClient
    _model: str
    _ratelimiter: ratelimit.RateLimiter

    def __init__(self, *, client: openai.AsyncClient, model: str, ratelimiter: ratelimit.RateLimiter) -> None:
        """Initializes an `AsyncLLMBackend` instance.

        Args:
            client: The OpenAI async client.
            model: The model name.
            ratelimiter: The rate limiter.
        """
        self._client = client
        self._model = model
        self._ratelimiter = ratelimiter

    async def __call__(self, *, response_format: type | None = None, **kwargs: Any) -> types.ModelResponse:
        """Calls the LLM backend and returns its output.

        Args:
            response_format: `The type of the expected response from the model.`
            **kwargs: Additional keyword arguments to pass to the API call.

        Returns:
            A `types.ModelResponse` object representing the model's response.
        """
        with self._ratelimiter:
            if response_format is not None:
                kwargs["response_format"] = response_format
                fn = self._client.chat.completions.parse
            else:
                fn = self._client.chat.completions.create
            return await fn(model=self._model, **kwargs)

    async def generate(self, chat: chat_lib.Chat, /, **kwargs: Any) -> types.ModelResponse:
        """Generates a response from the LLM asynchronously based on the chat history.

        Args:
            chat: A `chat_lib.Chat` object containing the conversation history.
            **kwargs: Additional keyword arguments to pass to the `__call__` method.

        Returns:
            A `types.ModelResponse` object representing the model's response.

        """
        return await self(messages=chat.messages, **kwargs)


@dataclasses.dataclass(kw_only=True)
class LLMBackendConfig:
    """Abstract base class for all LLM backends.

    This class defines the common interface and attributes expected from any LLM backend and ways to instantiate
    matching clients using the `openai` library
    """

    name: str
    """A human-readable name for the backend."""
    base_url: str
    """The base URL for the LLM API endpoint."""
    model_name: str
    """The specific model identifier to be used with this backend."""
    api_key: str | None = None
    """The API key required for authentication with the LLM service (optional)."""
    ratelimit: float | None = None

    def get_backend(self) -> LLMBackend:
        """Returns an `LLMBackend` instance configured for this backend configuration."""
        client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        ratelimiter = ratelimit.RateLimiter(self.ratelimit) if self.ratelimit is not None else None
        return LLMBackend(client=client, model=self.model_name, ratelimiter=ratelimiter)

    def get_async_backend(self) -> AsyncLLMBackend:
        """Returns an `AsyncLLMBackend` instance configured for this backend configuration."""
        client = openai.AsyncClient(base_url=self.base_url, api_key=self.api_key)
        ratelimiter = ratelimit.RateLimiter(self.ratelimit) if self.ratelimit is not None else None
        return AsyncLLMBackend(client=client, model=self.model_name, ratelimiter=ratelimiter)

    def get_async_client(self) -> tuple[openai.AsyncClient, str]:
        """Returns an asynchronous OpenAI client configured for this backend and the model name.

        This method abstracts the client creation, allowing different backends to use
        the same `openai.AsyncClient` interface, which is compatible with many LLM APIs.

        Returns:
            tuple[openai.AsyncClient, str]: A tuple containing the configured
                                            `openai.AsyncClient` instance and the model name.
        """
        client = openai.AsyncClient(base_url=self.base_url, api_key=self.api_key)
        return client, self.model_name

    def get_client(self) -> tuple[openai.Client, str]:
        """Returns a synchronous OpenAI client configured for this backend and the model name.

        This method abstracts the client creation, allowing different backends to use
        the same `openai.Client` interface, which is compatible with many LLM APIs.

        Returns:
            tuple[openai.Client, str]: A tuple containing the configured
                                       `openai.Client` instance and the model name.
        """
        client = openai.Client(base_url=self.base_url, api_key=self.api_key)
        return client, self.model_name


@dataclasses.dataclass(kw_only=True)
class Gemini2p5FlashLite(LLMBackendConfig):
    """Concrete implementation of LLMBackend for the Google Gemini API."""
    name: str = "Gemini 2.5 Flash Lite"
    base_url: str = _GOOGLE_OPENAI_API_BASE_URL
    model_name: str = "gemini-2.5-flash-lite"
    api_key: str | None = os.environ.get(_GOOGLE_API_KEY_ENV_VAR, None)
    ratelimit: float | None = 15.


@dataclasses.dataclass(kw_only=True)
class Gemini2p5Flash(LLMBackendConfig):
    """Concrete implementation of LLMBackend for the Google Gemini API."""
    name: str = "Gemini 2.5 Flash"
    base_url: str = _GOOGLE_OPENAI_API_BASE_URL
    model_name: str = "gemini-2.5-flash"
    api_key: str | None = os.environ.get(_GOOGLE_API_KEY_ENV_VAR, None)
    ratelimit: float | None = 10.


@dataclasses.dataclass(kw_only=True)
class Gemini2p5Pro(LLMBackendConfig):
    """Concrete implementation of LLMBackend for the Google Gemini API."""
    name: str = "Gemini 2.5 Pro"
    base_url: str = _GOOGLE_OPENAI_API_BASE_URL
    model_name: str = "gemini-2.5-pro"
    api_key: str | None = os.environ.get(_GOOGLE_API_KEY_ENV_VAR, None)
    ratelimit: float | None = 2.


@dataclasses.dataclass(kw_only=True)
class GPT5(LLMBackendConfig):
    """Concrete implementation of LLMBackend for the OpenAI GPT-5 API."""
    name = "GPT-5"
    base_url = _OPENAI_API_BASE_URL
    model_name: str = "gpt-5"
    api_key: str | None = os.environ.get(_OPENAI_API_KEY_ENV_VAR, None)


@dataclasses.dataclass(kw_only=True)
class LLamaCpp(LLMBackendConfig):
    """Concrete implementation of LLMBackend for llama.cpp servers."""
    name: str = "llama.cpp"
    model_name: str = "model"
    api_key: str | None = os.environ.get(_LLAMA_CPP_API_KEY_ENV_VAR, "sk-")


BACKENDS = {Gemini2p5FlashLite, Gemini2p5Flash, Gemini2p5Pro, GPT5, LLamaCpp}
BACKENDS_ENTRY = {backend.name: backend for backend in BACKENDS}
