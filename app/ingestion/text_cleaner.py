"""Conservative text-cleaning helpers for extracted document text."""

import re


def clean_text(text: str) -> str:
    """Clean extracted text while preserving its wording and paragraphs.

    Single line breaks are treated as line wrapping inside a paragraph. Blank
    lines remain paragraph separators.
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\x00", "")

    paragraphs: list[str] = []
    current_lines: list[str] = []

    for line in normalized.split("\n"):
        cleaned_line = re.sub(r"[ \t]+", " ", line).strip()

        if cleaned_line:
            current_lines.append(cleaned_line)
        elif current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines = []

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return "\n\n".join(paragraphs).strip()
