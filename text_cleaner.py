from __future__ import annotations

import re

# Common thinking/CoT tag patterns emitted by reasoning models
_THINK_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reason>.*?</reason>", re.DOTALL | re.IGNORECASE),
    re.compile(r"</think>", re.IGNORECASE),
    re.compile(r"<think>", re.IGNORECASE),
    re.compile(r"</thinking>", re.IGNORECASE),
    re.compile(r"<thinking>", re.IGNORECASE),
    re.compile(r"</reasoning>", re.IGNORECASE),
    re.compile(r"<reasoning>", re.IGNORECASE),
]


def strip_thinking_tags(text: str) -> str:
    """Remove thinking/chain-of-thought tags from model output."""
    if not text:
        return text

    cleaned = text
    for pattern in _THINK_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    # Collapse extra blank lines left behind
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()
