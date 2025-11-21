"""Utility functions for IPython/Jupyter environments."""
import asyncio
import difflib
import html
import pathlib
import traceback
from collections.abc import Callable, Sequence

import ipywidgets
from IPython import display as ipydisplay

from hslu.dlm03.common import agent as agent_lib
from hslu.dlm03.common import chat as chat_lib
from hslu.dlm03.common import types
from hslu.dlm03.common.displays import ipython_display
from hslu.dlm03.tools import lint
from hslu.dlm03.util import unified_diff


def display_agent(agent: agent_lib.Agent, chat: chat_lib.Chat | None = None) -> None:
    if chat is None:
        chat = chat_lib.Chat()
    display = ipython_display.IPythonChatDisplay()
    display.show()
    chat.add_observer(display)
    user_input = ipywidgets.Textarea(continuous_update=False, layout={"width": "75%", "height": "50px"})
    send_button = ipywidgets.Button(icon="send", layout={"height": "50px", "width": "75px"})

    def submit_fn(_):
        message = user_input.value
        user_input.value = ""
        chat.append(types.UserMessage(role="user", content=message))
        asyncio.run(agent(chat=chat))

    send_button.on_click(submit_fn)

    box_layout = ipywidgets.Layout(display="flex", align_items="center", width="100%")
    box = ipywidgets.HBox([user_input, send_button], layout=box_layout)
    ipydisplay.display(box)


def display_issues(
        widget: ipywidgets.HTML,
        content: str,
        issue: lint.Issue,
        fix: unified_diff.UnifiedDiff,
        num_lines: int | None = 3,
        strict: bool = True,
):
    """Generates and displays a side-by-side HTML diff in a Jupyter Notebook.

    Highlights changes and annotates the original issue location.
    """
    original_lines = content.splitlines()
    new_lines = fix(content, strict=strict).splitlines()
    if num_lines is None:
        num_lines = max(len(original_lines), len(new_lines))
    issue_rows = set(range(issue.location.row, issue.end_location.row + 1))
    matcher = difflib.SequenceMatcher(a=original_lines, b=new_lines)

    grouped_opcodes = matcher.get_grouped_opcodes(n=num_lines)

    html_rows = ['<thead><tr>'
                 '<th style="width: 40px; text-align: right; padding-right: 5px;"></th>'
                 '<th style="text-align: left;">Before</th>'
                 '<th style="width: 40px; text-align: right; padding-right: 5px;"></th>'
                 '<th style="text-align: left;">After</th>'
                 '</tr></thead>']

    style = {
        "del": 'style="background-color: #ffe9e9;"',
        "add": 'style="background-color: #e9ffe9;"',
        "issue": 'style="background-color: #ffffd0; border-left: 3px solid #f0c000;"',
        "num": 'style="text-align: right; padding-right: 5px; color: #888; user-select: none;"',
        "code": 'style="margin: 0; padding: 0 5px; white-space: pre;"',
        "gap": 'style="color: #888; text-align: center; user-select: none;"',
    }

    last_a_line = 0
    html_rows.append("<tbody>")
    for group in grouped_opcodes:
        if group[0][1] > last_a_line:
            html_rows.append(
                f'<tr><td {style["gap"]} colspan="4">...</td></tr>',
            )
        last_a_line = group[-1][2]

        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for k in range(i1, i2):
                    line_num_a = k + 1
                    line_num_b = (j1 + (k - i1)) + 1
                    line_a = html.escape(original_lines[k].rstrip("\n"))

                    html_rows.append(
                        f'<tr>'
                        f'<td {style["num"]}>{line_num_a}</td>'
                        f'<td><pre {style["code"]}>{line_a}</pre></td>'
                        f'<td {style["num"]}>{line_num_b}</td>'
                        f'<td><pre {style["code"]}>{line_a}</pre></td>'
                        f'</tr>',
                    )

                    if line_num_a in issue_rows:
                        html_rows.append(
                            f'<tr>'
                            f'<td></td>'
                            f'<td {style["issue"]} colspan="3">'
                            f'<strong>[{issue.code}]</strong> {html.escape(issue.message)}'
                            f'</td>'
                            f'</tr>',
                        )

            if tag == "delete" or tag == "replace":
                for k in range(i1, i2):
                    line_num_a = k + 1
                    line_a = html.escape(original_lines[k].rstrip("\n"))

                    html_rows.append(
                        f'<tr>'
                        f'<td {style["num"]}>{line_num_a}</td>'
                        f'<td {style["del"]}><pre {style["code"]}>{line_a}</pre></td>'
                        f'<td {style["num"]}></td>'
                        f'<td {style["del"]}></td>'
                        f'</tr>',
                    )

                    if line_num_a in issue_rows:
                        html_rows.append(
                            f'<tr>'
                            f'<td></td>'
                            f'<td {style["issue"]} colspan="3">'
                            f'<strong>[{issue.code}]</strong> {html.escape(issue.message)}'
                            f'</td>'
                            f'</tr>',
                        )

            if tag == "insert" or tag == "replace":
                for k in range(j1, j2):
                    line_num_b = k + 1
                    line_b = html.escape(new_lines[k].rstrip("\n"))

                    html_rows.append(
                        f'<tr>'
                        f'<td {style["num"]}></td>'
                        f'<td {style["add"]}></td>'
                        f'<td {style["num"]}>{line_num_b}</td>'
                        f'<td {style["add"]}><pre {style["code"]}>{line_b}</pre></td>'
                        f'</tr>',
                    )

    html_rows.append("</tbody>")
    final_html = (
            '<table style="font-family: monospace; width: 100%; border-collapse: collapse;">'
            + "\n".join(html_rows)
            + "</table>"
    )
    widget.value = final_html
    return final_html


def display_autofix(issue_generator: Callable[[], Sequence[lint.Issue]],
                    auto_fix: Callable[[lint.Issue], unified_diff.UnifiedDiff], num_lines: int | None,
                    strict: bool = True) -> None:
    """Displays side-by-side issues and generated auto-fixes in IPython.

    Args:
        issue_generator: A callable that returns a sequence of lint issues.
        auto_fix: A callable that takes a lint issue and returns a unified diff.
        num_lines: The number of context lines to display around changes in the diff.
        strict: Whether to apply the fix strictly (`True` by default) or to try to find the hunks in the file.
    """
    issue_widget = ipywidgets.HTML()
    accept = ipywidgets.Button(icon="check")
    discard = ipywidgets.Button(icon="times")
    box = ipywidgets.HBox([accept, discard])
    issues = issue_generator()
    ipydisplay.display(issue_widget, box)
    issue = None
    fix = None

    def next_issue() -> None:
        nonlocal issue
        nonlocal fix
        if issues:
            issue = issues.pop(0)
            path = pathlib.Path(issue.filename)
            source_code = path.read_text()
            try:
                fix = auto_fix(issue)
                display_issues(issue_widget, source_code, issue, fix, num_lines, strict)
            except Exception as e:  # noqa: BLE001
                traceback.print_exception(e)
                next_issue()
        else:
            ipydisplay.clear_output()

    def accept_fn(_: ipywidgets.Button) -> None:
        nonlocal issues
        nonlocal fix
        if fix:
            path = pathlib.Path(issue.filename)
            unified_diff.apply(fix, path, path, strict=False)
            issues = [i for i in issues if i.filename != issue.filename] + lint.lint(issue.filename)
        issue_widget.value = ""
        fix = None
        next_issue()

    def discard_fn(_: ipywidgets.Button) -> None:
        nonlocal fix
        issue_widget.value = ""
        fix = None
        next_issue()

    accept.on_click(accept_fn)
    discard.on_click(discard_fn)

    next_issue()
