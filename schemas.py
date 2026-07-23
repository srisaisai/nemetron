from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from config import settings


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class FunctionDefinition(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str = settings.nemetron_model
    messages: list[ChatMessage]
    tools: list[ToolDefinition] | None = None
    temperature: float | None = 0.7
    max_tokens: int | None = None
    stream: bool = False
    stop: list[str] | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    # Nemetron proxy extension: "agent" (default) or "passthrough"
    tool_mode: Literal["agent", "passthrough"] | None = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "nemetron"
