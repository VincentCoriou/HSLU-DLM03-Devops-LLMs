"""Provides classes for parsing and applying unified diff patches."""
import dataclasses
import difflib
import pathlib
import re
from collections.abc import Sequence

_UNIFIED_DIFF_HEADER_REGEX = re.compile(
    r"(?P<type>---|\+\+\+) (?P<file>[^\t]*)(?:\t(?P<timestamp>.*))?",
)
_UNIFIED_DIFF_HUNK_HEADER_REGEX = re.compile(
    r"@@ -(?P<from_line>\d+)(?:,(?P<from_count>\d+))? \+(?P<to_line>\d+)(?:,(?P<to_count>\d+))? @@(?: (?:.*))?",
)


class UnifiedDiffError(Exception):
    """Custom exception for unified diff parsing and applying errors."""


@dataclasses.dataclass
class UnifiedDiffHunk:
    """Represents a single hunk in a unified diff patch."""
    from_line: int
    """The starting line number in the original file."""
    from_count: int
    """The number of lines in the hunk in the original file."""
    to_line: int
    """The starting line number in the new file."""
    to_count: int
    """The number of lines in the hunk in the new file."""
    before: list[str]
    """The lines in the hunk from the original file."""
    after: list[str]
    """The lines in the hunk from the new file."""

    @classmethod
    def from_lines(cls, lines: Sequence[str]) -> "UnifiedDiffHunk":
        """Creates a UnifiedDiffHunk from a sequence of lines.

        Args:
            lines: A sequence of lines representing the hunk.

        Returns:
            A UnifiedDiffHunk object.

        Raises:
            ValueError: If the hunk header is invalid.
            UnifiedDiffError: If a hunk line has an invalid prefix.
        """
        header, lines = lines[0], lines[1:]
        match = _UNIFIED_DIFF_HUNK_HEADER_REGEX.match(header)
        if not match:
            error_message = f"Invalid unified diff hunk header: {header}"
            raise ValueError(error_message)
        from_line = int(match.group("from_line"))
        from_count = (
            int(match.group("from_count"))
            if match.group("from_count") is not None
            else 1
        )
        to_line = int(match.group("to_line"))
        to_count = (
            int(match.group("to_count")) if match.group("to_count") is not None else 1
        )
        before = []
        after = []
        for line in lines:
            prefix, content = line[0], line[1:]
            match prefix:
                case " ":
                    before.append(content)
                    after.append(content)
                case "+":
                    after.append(content)
                case "-":
                    before.append(content)
                case _:
                    error_message = f"Invalid unified diff hunk line prefix: {prefix}"
                    raise UnifiedDiffError(error_message)
        return cls(from_line, from_count, to_line, to_count, before, after)

    @classmethod
    def from_string(cls, string: str) -> "UnifiedDiffHunk":
        """Creates a UnifiedDiffHunk from a string.

        Args:
            string: A string representing the hunk.

        Returns:
            A UnifiedDiffHunk object.
        """
        lines = string.splitlines()
        return cls.from_lines(lines)

    def verify(self, content: Sequence[str], offset: int = 0) -> None:
        """Verifies that the hunk can be applied to the given content.

        Args:
            content: The content to verify against.
            offset: The line offset to apply.

        Raises:
            UnifiedDiffError: If the hunk cannot be applied.
        """
        if len(self.before) != self.from_count:
            error_message = (f"Invalid unified diff hunk: expected {self.from_count} lines before, "
                             f"got {len(self.before)}")
            raise UnifiedDiffError(error_message)
        if len(self.after) != self.to_count:
            error_message = f"Invalid unified diff hunk: expected {self.to_count} lines after, got {len(self.after)}"
            raise UnifiedDiffError(error_message)

        start_index = self.from_line
        if self.from_count > 0:
            start_index -= 1

        start = start_index + offset
        end = start + self.from_count

        before_diff = "\n".join(
            difflib.unified_diff(
                content[start:end],
                self.before,
                n=0,
                fromfile="original",
                tofile="expected",
                lineterm="",
            ),
        )
        if before_diff:
            error_message = ("Cannot apply unified diff on given content, original content does not match expected "
                             f"unified diff content:\n{before_diff}")
            raise UnifiedDiffError(error_message)

    def apply(self, content: list[str], offset: int = 0) -> int:
        """Applies the hunk to the given content.

        Args:
            content: The content to apply the hunk to.
            offset: The line offset to apply.

        Returns:
            The new offset after applying the hunk.
        """
        self.verify(content, offset)
        start_index = self.from_line
        if self.from_count > 0:
            start_index -= 1

        start = start_index + offset
        end = start + self.from_count

        content[start:end] = self.after
        return self.to_count - self.from_count

    def find(self, lines: Sequence[str]) -> "UnifiedDiffHunk":
        """Finds the hunk in the given content.

        Args:
            lines: The content to find the hunk in.

        Returns:
            The hunk in the given content.

        Raises:
            UnifiedDiffError: If the hunk cannot be found.
        """
        matcher = difflib.SequenceMatcher(None, lines, self.before)
        match = matcher.find_longest_match(0, len(lines), 0, len(self.before))
        if match.size != len(self.before):
            error_message = "Could not find hunk in content."
            raise UnifiedDiffError(error_message)
        offset = (match.a + 1) - self.from_line
        return UnifiedDiffHunk(self.from_line + offset, len(self.before), self.to_line + offset, len(self.after),
                               self.before, self.after)


@dataclasses.dataclass()
class UnifiedDiff:
    """Represents a UnifiedDiff patch."""
    from_file: str
    """The path to the original file."""
    to_file: str
    """The path to the new file."""
    hunks: list[UnifiedDiffHunk]
    """A list of hunks in the patch."""

    @classmethod
    def from_string(cls, string: str) -> "UnifiedDiff":
        """Creates a UnifiedDiff from a string.

        Args:
            string: A string representing the unified diff patch.

        Returns:
            A UnifiedDiff object.

        Raises:
            UnifiedDiffError: If the unified diff header is invalid.
        """
        lines = string.splitlines()
        header, lines = lines[:2], lines[2:]
        from_file = None
        to_file = None
        for line in header:
            match = _UNIFIED_DIFF_HEADER_REGEX.match(line)
            if not match:
                error_message = f"Invalid unified diff header: {line}"
                raise UnifiedDiffError(error_message)
            header_type = match.group("type")
            match header_type:
                case "---":
                    from_file = match.group("file")
                case "+++":
                    to_file = match.group("file")
                case _:
                    error_message = f"Invalid unified diff header type: {header_type}"
                    raise UnifiedDiffError(error_message)
        hunks = []
        last = None
        for i, line in enumerate(lines):
            if _UNIFIED_DIFF_HUNK_HEADER_REGEX.match(line):
                if last is not None:
                    hunk = UnifiedDiffHunk.from_lines(lines[last:i])
                    hunks.append(hunk)
                last = i
        if last is not None:
            hunk = UnifiedDiffHunk.from_lines(lines[last:])
            hunks.append(hunk)
        return cls(from_file, to_file, hunks)

    def apply(self, content: list[str], *, strict: bool = True) -> list[str]:
        """Applies the patch to the given content.

        Args:
            content: The content to apply the patch to, as a list of lines.
            strict: Whether to apply the patch strictly (`True` by default) or to try to find the hunks in the file.

        Returns:
            The patched content as a list of lines.
        """
        offset = 0
        hunks = self.hunks if strict else [hunk.find(content) for hunk in self.hunks]
        for hunk in hunks:
            offset += hunk.apply(content, offset)
        return content

    def __call__(self, content: str, *, strict: bool = True) -> str:
        """Applies the patch to the given content as a string.

        Args:
            content: The content to apply the patch to, as a string.
            strict: Whether to apply the patch strictly (`True` by default) or to try to find the hunks in the file.

        Returns:
            The patched content as a string.
        """
        return "\n".join(self.apply(content.splitlines(), strict=strict))


def apply(diff: UnifiedDiff, from_file: pathlib.Path | None = None, to_file: pathlib.Path | None = None, *,
          strict: bool = True) -> None:
    """Applies a unified diff patch to a file.

    Args:
        diff: The unified diff patch to apply.
        from_file: The path to the original file.
        to_file: The path to the new file.
        strict: Whether to apply the patch strictly (`True` by default) or to try to find the hunks in the file.
    """
    if from_file is None:
        from_file = pathlib.Path(diff.from_file)
    if to_file is None:
        to_file = pathlib.Path(diff.to_file)
    original_content = from_file.read_text()
    patched_content = diff(original_content, strict=strict)
    to_file.write_text(patched_content)
