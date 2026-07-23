from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from schemas import ChatMessage


def openai_message_to_langchain(msg: ChatMessage) -> BaseMessage:
    """Convert an OpenAI-style chat message to a LangChain message."""
    if msg.role == "system":
        return SystemMessage(content=msg.content or "")
    if msg.role == "user":
        return HumanMessage(content=msg.content or "")
    if msg.role == "assistant":
        return AIMessage(content=msg.content or "", tool_calls=msg.tool_calls or [])
    if msg.role == "tool":
        return ToolMessage(content=msg.content or "", tool_call_id=msg.tool_call_id or "")
    return HumanMessage(content=msg.content or "")
