from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def message_to_openai(msg: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message to OpenAI chat format."""
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, HumanMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, AIMessage):
        openai_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            openai_msg["tool_calls"] = [
                {
                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            json.dumps(tc.get("args", {}))
                            if isinstance(tc.get("args"), dict)
                            else str(tc.get("args", "{}"))
                        ),
                    },
                }
                for tc in msg.tool_calls
            ]
        return openai_msg
    if isinstance(msg, ToolMessage):
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
        }
    return {"role": "user", "content": str(msg.content)}


def openai_choice_to_ai_message(choice: dict[str, Any]) -> AIMessage:
    """Convert an OpenAI completion choice into an AIMessage."""
    message = choice.get("message", {})
    content = message.get("content") or ""
    tool_calls = []

    for tc in message.get("tool_calls", []):
        function_info = tc.get("function", {})
        args = function_info.get("arguments", "{}")
        try:
            parsed_args = json.loads(args) if isinstance(args, str) else args
        except json.JSONDecodeError:
            parsed_args = {"raw": args}
        tool_calls.append(
            {
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": function_info.get("name", ""),
                "args": parsed_args,
            }
        )

    return AIMessage(content=content, tool_calls=tool_calls)
