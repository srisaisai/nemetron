from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator


def _sse_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def build_openai_response(
    content: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Build a non-streaming OpenAI-compatible chat completion response."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def build_tool_calls_response(
    ai_message,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict[str, Any]:
    """Build an OpenAI-compatible response containing raw tool_calls (pass-through mode)."""
    from message_utils import message_to_openai

    openai_msg = message_to_openai(ai_message)
    finish_reason = "tool_calls" if getattr(ai_message, "tool_calls", None) else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": openai_msg,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def stream_openai_response(
    content: str,
    model: str,
) -> AsyncIterator[str]:
    """Stream an OpenAI-compatible chat completion response."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield _sse_event({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    chunk_size = 8
    for i in range(0, len(content), chunk_size):
        part = content[i : i + chunk_size]
        yield _sse_event({
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
        })

    yield _sse_event({
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    })
    yield "data: [DONE]\n\n"

