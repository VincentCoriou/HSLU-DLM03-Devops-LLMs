"""Provides an agent abstraction."""

import random
from collections.abc import Callable, Sequence
from typing import Any

from hslu.dlm03.common import backend as backend_lib
from hslu.dlm03.common import chat as chat_lib
from hslu.dlm03.common import tools, types


class Agent:
    """A class used to represent an AI Agent."""
    _backend: backend_lib.AsyncLLMBackend
    _tool_manager: tools.ToolManager

    def __init__(self, backend: backend_lib.AsyncLLMBackend, tool_manager: tools.ToolManager | None = None) -> None:
        """Initializes an `Agent` instance.

        Args:
            backend: The backend to use.
            tool_manager: The tool manager to use (optional).
        """
        self._backend = backend
        self._tool_manager = tool_manager

    async def tools(self) -> Sequence[types.Tool]:
        """Returns the list of tools that this agent can use.

        Returns:
            A list of `types.Tool` objects representing the available tools.
        """
        if self._tool_manager is None:
            return []
        return await self._tool_manager.tools()

    async def __call__(self, chat: chat_lib.Chat, **kwargs: Any) -> list[types.Message]:
        """Calls the agent.

        Args:
            chat: The chat to use.
            **kwargs: Additional keyword arguments to pass to the backend.

        Returns:
            A list of `types.Message` objects representing the agent's response.
        """
        tools = await self.tools()
        done = False
        input_length = len(chat.messages)
        while not done:
            response = await self._backend.generate(chat, tools=tools, **kwargs)
            choice: types.Choice = random.choice(response.choices)  # noqa: S311
            message = choice.message
            chat.append(message)
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_call_output = await self._tool_manager(tool_call)
                    chat.append(*tool_call_output)
                    done = False
            else:
                done = True
        return chat.messages[input_length:]


async def agent_loop(input_fn: Callable[[], str], /, *, agent: Agent, chat: chat_lib.Chat) -> chat_lib.Chat:
    """Runs an agent loop.

    Args:
        input_fn: A callable that takes no arguments and returns a string.
        agent: The agent to use.
        chat: The chat to use.

    Returns:
        The chat after the agent loop has finished.
    """
    while True:
        request = input_fn()
        if not request:
            break
        chat.append(types.UserMessage(role="user", content=request))
        await agent(chat=chat)
    return chat
