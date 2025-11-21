import json
import subprocess
from collections.abc import Sequence

import dataclass_wizard
from pydantic import dataclasses

from hslu.dlm03.util import unified_diff


@dataclasses.dataclass
class Location:
    """Represents a specific location within a file, identified by column and row.

    Attributes:
        column: The column number (1-indexed).
        row: The row number (1-indexed).
    """
    column: int
    row: int


@dataclasses.dataclass
class Edit:
    """Represents a single textual edit to be applied to a file.

    Attributes:
        content: The new content to insert or replace with.
        location: The starting location of the edit.
        end_location: The ending location of the edit.
    """
    content: str
    location: Location
    end_location: Location

    def to_unified_diff_hunk(self, lines: Sequence[str]) -> unified_diff.UnifiedDiffHunk:
        return unified_diff.UnifiedDiffHunk(
            from_line=self.location.row,
            from_count=self.end_location.row - self.location.row + 1,
            to_line=self.location.row,
            to_count=len(self.content.splitlines()),
            before=lines[self.location.row - 1: self.end_location.row],
            after=self.content.splitlines(),
        )


@dataclasses.dataclass
class Fix:
    """Represents a suggested fix for an issue, consisting of one or more edits.

    Attributes:
        applicability: A string indicating how applicable the fix is (e.g., "always", "sometimes").
        edits: A sequence of `Edit` objects that constitute the fix.
        message: A human-readable message describing the fix.
    """
    applicability: str
    edits: Sequence[Edit]
    message: str

    def to_unified_diff(self, filename: str, lines: Sequence[str]) -> unified_diff.UnifiedDiff:
        hunks = [edit.to_unified_diff_hunk(lines) for edit in self.edits]
        return unified_diff.UnifiedDiff(
            from_file=f"a/{filename}", to_file=f"b/{filename}", hunks=hunks)


@dataclasses.dataclass
class Issue:
    """Represents a single linting reported issue.

    Attributes:
        filename: The name of the file where the issue was found.
        cell: The cell number if the issue is in a notebook, otherwise None.
        code: The unique code identifying the type of issue (e.g., "E123").
        message: A human-readable description of the issue.
        location: The starting location of the issue.
        end_location: The ending location of the issue.
        noqa_row: The row number where a '# noqa' comment could be placed to ignore this issue.
        url: An optional URL providing more information about the issue.
        fix: An optional `Fix` object suggesting how to resolve the issue.
    """
    filename: str
    cell: int | None
    code: str
    message: str
    location: Location
    end_location: Location
    noqa_row: int
    url: str | None
    fix: Fix | None

    def fix_to_unified_diff(self, content: str) -> unified_diff.UnifiedDiff:
        if self.fix is None:
            return unified_diff.UnifiedDiff(from_file=f"a/{self.filename}", to_file=f"b/{self.filename}", hunks=[])
        return self.fix.to_unified_diff(self.filename, content.splitlines())


@dataclasses.dataclass
class Issues:
    issues: list[Issue]


def lint(path: str) -> list[Issue]:
    """Lints a Python file using ruff and returns a list of found issues.

    Args:
        path: The path to the file to lint.

    Returns:
        A list of `Issue` objects, each representing a linting issue.

    Raises:
        RuntimeError: If the ruff process itself fails (e.g., due to configuration issues,
                      or if ruff returns an unexpected exit code other than 0 or 1).
    """
    process = subprocess.run(
        [f"ruff check --output-format=json {path}"],
        capture_output=True, shell=True,
        check=False,
    )
    if process.returncode not in {0, 1}:
        raise RuntimeError(process.stderr.decode("utf-8"))
    data = json.loads(process.stdout)
    return dataclass_wizard.fromlist(Issue, data)
