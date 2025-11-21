"""Provides an IPython display for chat messages."""
import html
import json
import typing
from xml.etree import ElementTree as ET

import ipywidgets
import markdown
from IPython import display as ipydisplay

from hslu.dlm03.common import chat_display, types

_DEFAULT_CSS = """
.chat-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: #ffffff;
    padding: 15px;
    border-radius: 12px;

    border: none;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);

    max-width: 700px;
    margin: 20px auto;
    box-sizing: border-box;

    display: flex;
    flex-direction: column;
    gap: 12px;
}

.message {
    padding: 12px 18px;
    border-radius: 18px;
    max-width: 75%;
    word-wrap: break-word;
    line-height: 1.5;

    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.user {
    background-color: #007aff;
    color: white;
    align-self: flex-end;

    border-bottom-right-radius: 4px;
}

.assistant {
    background-color: #f1f1f1;
    color: #1c1c1e;
    align-self: flex-start;

    border-bottom-left-radius: 4px;
}

.system {
    color: #666;
    font-size: 0.9em;
    font-style: italic;
    text-align: center;
    max-width: 100%;
    align-self: center;

    background-color: transparent;
    box-shadow: none;
}

.tool_call {
    font-family: 'SFMono-Regular', Consolas, 'Courier New', monospace;
    font-size: 0.9em;

    color: #855300;
    background-color: #fff8e1;
    border: 1px solid #ffe0b2;

    border-radius: 6px;

    align-self: flex-start;
    text-align: left;
    box-shadow: none;
}

.tool {
    font-family: 'SFMono-Regular', Consolas, 'Courier New', monospace;
    font-size: 0.9em;

    color: #1b5e20;
    background-color: #e8f5e9;
    border: 1px solid #b2f5d4;

    border-radius: 6px;

    align-self: flex-end;
    text-align: left;
    box-shadow: none;
}
"""


class IPythonChatDisplay(chat_display.ChatDisplay):
    """An IPython display for chat messages."""

    _widget: ipywidgets.HTML
    _html: ET.Element
    _css: ET.Element

    def __init__(self, *, css: str = _DEFAULT_CSS) -> None:
        """Initializes an `IPythonDisplay` instance.

        Args:
            css: The CSS to apply to the IPython widget.
        """
        super().__init__()
        self._widget = ipywidgets.HTML()

        self._css = ET.Element("style")
        self._css.text = css

        self._html = ET.Element("div")
        self.clear()

    def show(self) -> None:
        """Displays the chat widget in the IPython environment."""
        ipydisplay.display(self._widget)

    def reload(self) -> None:
        """Reloads the current HTML content into the widget."""
        self._widget.value = ET.tostring(self._css).decode() + ET.tostring(self._html).decode()

    def _add_message(self, role: str, text: str) -> None:
        """Adds a message to the chat.

        Args:
            role: The role of the message.
            text: The text of the message.
        """
        element = ET.Element("div", attrib={"class": f"message {role}"})
        try:
            message = ET.fromstring("<div>" + markdown.markdown(text) + "</div>")  # noqa: S314
        except Exception: # noqa: BLE001
            message = ET.fromstring("<div>" + html.escape(text) + "</div>")
        element.append(message)

        self._html.append(element)
        self.reload()

    @typing.override
    def clear(self) -> None:
        self._html.clear()
        self._html.attrib["class"] = "chat-container"
        self._html.attrib["id"] = "chat"

        self.reload()

    @typing.override
    def display_system(self, message: types.SystemMessage) -> None:
        self._add_message(self.role(message), self.content(message))

    @typing.override
    def display_user(self, message: types.UserMessage) -> None:
        self._add_message(self.role(message), self.content(message))

    @typing.override
    def display_assistant(self, message: types.AssistantMessage) -> None:
        self._add_message(self.role(message), self.content(message))

    @typing.override
    def display_tool_call(self, tool_call: types.ToolCall) -> None:
        arguments = json.loads(tool_call.function.arguments)
        arguments_string = f"{', '.join([f'{k}={v}' for k, v in arguments.items()])}"
        tool_call_string = f"{tool_call.function.name}({arguments_string})"
        self._add_message("tool_call", tool_call_string)

    @typing.override
    def display_tool_call_output(self, message: types.ToolCallOutput) -> None:
        self._add_message(self.role(message), self.content(message))
