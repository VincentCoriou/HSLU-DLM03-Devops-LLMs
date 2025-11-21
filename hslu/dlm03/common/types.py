"""Provides common types definitions."""

from typing import Any

from openai.types import chat
from openai.types.chat import (
    chat_completion,
    chat_completion_message_function_tool_call,
)
from openai.types.responses import response_input_param
from openai.types.shared_params import function_definition

SystemMessage = chat.ChatCompletionSystemMessageParam
AssistantMessage = chat.ChatCompletionMessage
UserMessage = chat.ChatCompletionUserMessageParam
ToolCall = (
    chat_completion_message_function_tool_call.ChatCompletionMessageFunctionToolCall
)
FunctionCall = chat_completion_message_function_tool_call.Function
ToolCallOutput = response_input_param.FunctionCallOutput
Message = SystemMessage | AssistantMessage | UserMessage | ToolCallOutput

ModelResponse = chat.ChatCompletion
Tool = chat.ChatCompletionToolParam
Function = function_definition.FunctionDefinition
Choice = chat_completion.Choice


def dict_to_message(**message: Any) -> Message:
    """Converts the given dictionary to a message.

    Args:
        **message: The dictionary to convert.

    Returns:
        A message object.
    """
    role = message.get("role")
    match role:
        case "system":
            return SystemMessage(**message)
        case "assistant":
            return AssistantMessage(**message)
        case "user":
            return UserMessage(**message)
        case "tool":
            return ToolCallOutput(**message)
        case _:
            error_message = f"Invalid message role: {role}"
            raise ValueError(error_message)


def message_to_dict(message: Message) -> dict[str, Any]:
    """Converts the given message to a dictionary.

    Args:
        message: The message to convert.

    Returns:
        A dictionary representation of the message.
    """
    return message.to_dict() if hasattr(message, "to_dict") else dict(message)
