from __future__ import annotations

import re

# Tag pairs: (opening_tag, closing_tag)
# Each pair is tried in order. For a given pair we:
#   1. Remove complete blocks  opening ... closing  (content included)
#   2. If an opening tag remains with no closing tag, remove from opening to END
#   3. If a closing tag remains with no opening tag, remove from START to closing
_THINK_TAG_PAIRS = [
    (r"<think>", r"</think>"),
    (r"<thinking>", r"</thinking>"),
    (r"<reasoning>", r"</reasoning>"),
    (r"<reason>", r"</reason>"),
    (r"<analysis>", r"</analysis>"),
    (r"<reflection>", r"</reflection>"),
    (r"<scratchpad>", r"</scratchpad>"),
]

_COMPILED_PAIRS = [
    (re.compile(op, re.IGNORECASE), re.compile(cl, re.IGNORECASE))
    for op, cl in _THINK_TAG_PAIRS
]


def strip_thinking_tags(text: str) -> str:
    """Remove thinking/chain-of-thought blocks (including content) from model output.

    Handles:
      - Complete blocks:  opening ... closing  -> removed entirely
      - Opening tag only (no closing):  opening ... END -> removed
      - Closing tag only (no opening):  START ... closing -> removed
      - Stray leftover tags
    """
    if not text:
        return text

    cleaned = text

    for opening_re, closing_re in _COMPILED_PAIRS:
        # 1. Remove complete blocks (opening ... closing) including content
        block_pattern = re.compile(
            opening_re.pattern + r".*?" + closing_re.pattern,
            re.DOTALL | re.IGNORECASE,
        )
        cleaned = block_pattern.sub("", cleaned)

        # 2. Opening tag remains with no closing -> remove from opening to END
        if opening_re.search(cleaned) and not closing_re.search(cleaned):
            match = opening_re.search(cleaned)
            cleaned = cleaned[: match.start()]

        # 3. Closing tag remains with no opening -> remove from START to closing
        if not opening_re.search(cleaned) and closing_re.search(cleaned):
            match = closing_re.search(cleaned)
            cleaned = cleaned[match.end():]

        # 4. Remove any remaining stray tags of this pair
        cleaned = opening_re.sub("", cleaned)
        cleaned = closing_re.sub("", cleaned)

    # Collapse extra blank lines left behind
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()
