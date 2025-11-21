"""Provides an abstraction for displaying chat messages."""

import abc
import typing

from hslu.dlm03.common import chat, types


class ChatDisplay(abc.ABC, chat.ChatObserver):
    """Abstract base class for displaying chat messages."""

    @typing.override
    def update(self, message: types.Message) -> None:
        self.display(message)

    @abc.abstractmethod
    def clear(self) -> None:
        """Clears the display."""
        raise NotImplementedError

    @staticmethod
    def content(message: types.Message) -> str:
        """Returns the content of a message.

        Args:
            message: The message to extract the content from.

        Returns:
            The content of the message.
        """
        return message.content if hasattr(message, "content") else message["content"]

    @staticmethod
    def role(message: types.Message) -> str | None:
        """Returns the role of a message.

        Args:
            message: The message to extract the role from.

        Returns:
            The role of the message.
        """
        if hasattr(message, "role"):
            return message.role
        if "role" in message:
            return message["role"]
        return None

    def display(self, message: types.Message) -> None:
        """Displays a message.

        Args:
            message: The message to display.

        """
        role = self.role(message)
        match role:
            case "system":
                self.display_system(message)
            case "assistant":
                if message.content:
                    self.display_assistant(message)
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        self.display_tool_call(tool_call)
            case "user":
                self.display_user(message)
            case "tool":
                self.display_tool_call_output(message)
            case _:
                error_message = f"Unknown message role: {role}"
                raise ValueError(error_message)

    @abc.abstractmethod
    def display_system(self, message: types.SystemMessage) -> None:
        """Displays a system message.

        Args:
            message: The system message to display.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def display_user(self, message: types.UserMessage) -> None:
        """Displays a user message.

        Args:
            message: The user message to display.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def display_assistant(self, message: types.AssistantMessage) -> None:
        """Displays an assistant message.

        Args:
            message: The assistant message to display.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def display_tool_call(self, tool_call: types.ToolCall) -> None:
        """Displays a tool call.

        Args:
            tool_call: The tool call to display.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def display_tool_call_output(self, message: types.ToolCallOutput) -> None:
        """Displays a tool call output.

        Args:
            message: The tool call output to display.
        """
        raise NotImplementedError
