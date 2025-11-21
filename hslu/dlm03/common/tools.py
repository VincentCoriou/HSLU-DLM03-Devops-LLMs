"""Provides common types definitions for MCP."""

import contextlib
import json
from collections.abc import AsyncGenerator, Callable
from typing import Any

import mcp
from mcp import types as mcp_types
from mcp.client import streamable_http

from hslu.dlm03.common import types

ClientSession = mcp.ClientSession
ClientSessionFactory = Callable[[], contextlib.AbstractAsyncContextManager[ClientSession]]


@contextlib.asynccontextmanager
async def mcp_session(url: str, *,
                      authorization: str | None = None) -> \
        AsyncGenerator[mcp.ClientSession, None]:
    """Creates an MCP client session.

    Args:
        url: The URL of the MCP server.
        authorization: The authorization token to use.

    Yields:
        An MCP client session.
    """
    headers = {}
    if authorization:
        headers["Authorization"] = f"Bearer {authorization}"
    async with streamable_http.streamablehttp_client(url, headers=headers) as (
            read_stream,
            write_stream,
            _,
    ), mcp.ClientSession(read_stream, write_stream) as session:
        await session.initialize()
        yield session


def mcp_session_factory(url: str, **kwargs: Any) -> ClientSessionFactory:
    """Returns a factory for MCP client sessions.

    Args:
        url: The URL of the MCP server.
        **kwargs: Additional keyword arguments to pass to `mcp_session`.

    Returns:
        A callable that returns an asynchronous context manager for an MCP client session.
    """
    return lambda: mcp_session(url, **kwargs)


class ToolManager:
    """A class for managing MCP tools."""

    _session_factory: ClientSessionFactory
    _allowed_tools: set[str] | None

    def __init__(self, session_factory: ClientSessionFactory, allowed_tools: set[str] | None = None) -> None:
        """Initializes a `ToolManager` instance.

        Args:
            session_factory: A callable that returns an asynchronous context manager for an MCP client session.
            allowed_tools: A set of allowed tool names. If None, all tools are allowed.
        """
        self._session_factory = session_factory
        self._allowed_tools = allowed_tools

    async def available_tools(self) -> list[types.Tool]:
        """Retrieves a list of available tools from the MCP server.

        Returns:
            A list of `types.Tool` objects representing the available tools.
        """
        async with self._session_factory() as session:
            tools_response = await session.list_tools()
            return [tool_from_mcp(tool) for tool in tools_response.tools]

    async def tools(self) -> list[types.Tool]:
        """Returns the list of tools that this agent can use.

        Returns:
            A list of `types.Tool` objects representing the available tools.
        """
        tools = await self.available_tools()
        if self._allowed_tools:
            tools = [
                tool
                for tool in tools
                if tool["function"]["name"] in self._allowed_tools
            ]
        return tools

    async def __call__(self, tool_call: types.ToolCall) -> list[types.ToolCallOutput]:
        """Calls a tool and returns its output.

        Args:
            tool_call: The tool call to execute.

        Returns:
            A list of `types.ToolCallOutput` objects representing the tool's output.
        """
        tool_name = tool_call.function.name
        if self._allowed_tools is not None and tool_name not in self._allowed_tools:
            return [types.dict_to_message(role="user", call_id=tool_call.id, type="function_call_output",
                                          content=f"Cannot call tool {tool_name}.")]
        arguments = tool_call.function.arguments
        async with self._session_factory() as session:
            result = await session.call_tool(tool_name, json.loads(arguments))
            return [
                tool_call_result_from_mcp(
                    tool_call.id,
                    content,
                )
                for content in result.content
            ]

    @classmethod
    def from_url(cls, url: str, allowed_tools: set[str] | None = None, **kwargs: Any) -> "ToolManager":
        """Creates a `ToolManager` instance from a URL.

        Args:
            url: The URL of the MCP server.
            allowed_tools: A set of allowed tool names. If None, all tools are allowed.
            **kwargs: Additional keyword arguments to pass to `mcp_session`.

        Returns:
            A `ToolManager` instance.
        """
        return cls(mcp_session_factory(url, **kwargs), allowed_tools)


async def list_tools(session: mcp.ClientSession) -> list[types.Tool]:
    """Retrieves a list of available tools from an MCP server.

    Args:
        session: The MCP client session.

    Returns:
        A list of `types.Tool` objects representing the available tools.
    """
    tools_response = await session.list_tools()
    return [tool_from_mcp(tool) for tool in tools_response.tools]


async def get_tools(url: str, **kwargs: Any) -> list[types.Tool]:
    """Retrieves a list of available tools from an MCP server.

    Args:
        url: The URL of the MCP server.
        **kwargs: Additional keyword arguments to pass to `mcp_session`.

    Returns:
        A list of `types.Tool` objects representing the available tools.
    """
    async with mcp_session(url, **kwargs) as session:
        return await list_tools(session)


def tool_from_mcp(tool: mcp_types.Tool) -> types.Tool:
    """Converts the tool from `mcp` to `openai` compatible format.

    Args:
        tool: The MCP tool.

    Returns:
        types.Tool: The tool in `openai` compatible format.
    """
    return types.Tool(
        type="function",
        function=types.Function(
            name=tool.name,
            description=tool.description,
            parameters=tool.inputSchema,
            strict=True,
        ),
    )


def tool_call_result_from_mcp(call_id: str, content: mcp_types.ContentBlock) -> types.ToolCallOutput:
    """Converts the tool call result from `mcp` to `openai` compatible format.

    Args:
        call_id: str: The ID of the tool call.
        content: mcp_types.ContentBlock: The content block from the MCP tool call result.

    Returns:
        types.ToolCallOutput: The tool call output in `openai` compatible format.
    """
    content_type = content.type
    match content_type:
        case "text":
            return types.dict_to_message(
                role="tool",
                tool_call_id=call_id,
                content=content.text,
            )
        case "resource":
            resource = content.resource
            mime_type = resource.mimeType.split(";")[0]
            match mime_type:
                case "text/plain":
                    return types.dict_to_message(
                        role="tool",
                        tool_call_id=call_id,
                        content=resource.text,
                    )
                case _:
                    error_message = f"Unsupported resource mime type: {mime_type}"
                    raise ValueError(error_message)
        case _:
            error_message = f"Invalid content type: {content_type}"
            raise ValueError(error_message)
