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


class StreamThinkingFilter:
    """Streaming-safe remover for thinking/reasoning blocks.

    The non-streaming :func:`strip_thinking_tags` operates on a complete
    string. When the model output arrives token-by-token (streaming
    passthrough mode), a thinking tag such as ``<think>`` can be split
    across two chunks (e.g. ``"<thi"`` + ``"nk>..."``). This filter holds
    back just enough of the tail to detect a tag forming, discards the
    contents of thinking blocks, and emits only the safe content.

    Matching is case-insensitive, mirroring :func:`strip_thinking_tags`.
    """

    # Reuse the same tag pairs defined above for the non-streaming path.
    _OPENINGS: list[str] = [op for op, _cl in _THINK_TAG_PAIRS]
    _CLOSINGS: list[str] = [cl for _op, cl in _THINK_TAG_PAIRS]

    def __init__(self) -> None:
        self._buffer: str = ""
        self._in_thinking: bool = False
        # Lower-cased close tag we are currently hunting for, or None.
        self._close_tag: str | None = None

    # -- public API ---------------------------------------------------------
    def feed(self, chunk: str) -> str:
        """Append ``chunk``; return content that is safe to emit now."""
        if not chunk:
            return ""
        self._buffer += chunk
        out: list[str] = []
        while True:
            if not self._in_thinking:
                pos, _open, close_lower = self._find_opening(self._buffer)
                if pos is not None:
                    out.append(self._buffer[:pos])
                    self._buffer = self._buffer[pos + len(_open):]
                    self._in_thinking = True
                    self._close_tag = close_lower
                    continue
                # No complete opening tag. Hold back a tail that could be the
                # start of an opening tag so we do not leak a partial one.
                hold = self._partial_suffix(
                    self._buffer, [o.lower() for o in self._OPENINGS]
                )
                if hold:
                    safe = self._buffer[: len(self._buffer) - hold]
                    out.append(safe)
                    self._buffer = self._buffer[len(self._buffer) - hold:]
                else:
                    out.append(self._buffer)
                    self._buffer = ""
                break
            else:
                # Inside a thinking block: look for the matching close tag.
                idx = self._ci_find(self._buffer, self._close_tag)
                if idx is not None:
                    self._buffer = self._buffer[idx + len(self._close_tag):]
                    self._in_thinking = False
                    self._close_tag = None
                    continue
                # Discard thinking content but keep a tail that could be the
                # start of the close tag (so a split close still resolves).
                hold = self._partial_suffix(
                    self._buffer, [self._close_tag]
                )
                if hold and hold < len(self._buffer):
                    self._buffer = self._buffer[-hold:]
                elif hold == 0:
                    self._buffer = ""
                # If hold == len(buffer) the whole buffer is a strict prefix of
                # the close tag: keep it verbatim for the next feed().
                break
        return "".join(out)

    def flush(self) -> str:
        """Return any remaining safe content at end of stream."""
        if self._in_thinking:
            # Unterminated thinking block: drop everything.
            self._buffer = ""
            self._in_thinking = False
            self._close_tag = None
            return ""
        out = self._buffer
        self._buffer = ""
        return out

    # -- helpers ------------------------------------------------------------
    @classmethod
    def _find_opening(cls, text: str):
        """Return (position, actual-cased opening, lower-cased closing) of the
        earliest opening tag in ``text``, or ``(None, None, None)``."""
        lower = text.lower()
        best_pos: int | None = None
        best_open: str | None = None
        best_close: str | None = None
        for open_tag, close_tag in _THINK_TAG_PAIRS:
            idx = lower.find(open_tag)
            if idx == -1:
                continue
            if (
                best_pos is None
                or idx < best_pos
                or (idx == best_pos and len(open_tag) > len(best_open))
            ):
                best_pos = idx
                best_open = text[idx: idx + len(open_tag)]
                best_close = close_tag.lower()
        if best_pos is None:
            return None, None, None
        return best_pos, best_open, best_close

    @staticmethod
    def _ci_find(text: str, needle_lower: str) -> int | None:
        idx = text.lower().find(needle_lower)
        return idx if idx != -1 else None

    @staticmethod
    def _partial_suffix(text: str, needles_lower: list[str]) -> int:
        """Length of the longest suffix of ``text`` that is a *strict* prefix
        of any of ``needles_lower`` (i.e. shorter than the full needle)."""
        lower = text.lower()
        max_len = 0
        for needle in needles_lower:
            n = len(needle)
            upper_i = min(len(lower), n)
            for i in range(1, upper_i + 1):
                if i >= n:
                    break
                if needle.startswith(lower[-i:]) and i > max_len:
                    max_len = i
        return max_len
