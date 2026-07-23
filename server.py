from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse

from agent import NemetronAgent
from chat_model import NemetronChatModel
from config import settings
from message_conversion import openai_message_to_langchain
from response_builder import (
    build_openai_response,
    build_tool_calls_response,
    stream_openai_response,
)
from schemas import ChatCompletionRequest, ToolDefinition
from tools import get_tools

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title="Nemetron LangChain Proxy")


def filter_tools(
    incoming_tools: list[ToolDefinition] | None,
    allowed_names: set[str],
) -> list[ToolDefinition]:
    """Keep only tools that are in the curated allowlist."""
    if not incoming_tools:
        return []
    return [tool for tool in incoming_tools if tool.function.name in allowed_names]


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.nemetron_model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nemetron",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, http_request: FastAPIRequest):
    # Determine tool mode:
    #   - "agent" (default): execute tools internally, return final answer
    #   - "passthrough": return raw OpenAI-format tool_calls without executing
    tool_mode = (
        http_request.headers.get("x-tool-mode")
        or getattr(request, "tool_mode", None)
        or "agent"
    ).lower()
    logger.info(
        "Incoming chat completion request (stream=%s, tool_mode=%s)",
        request.stream,
        tool_mode,
    )

    explicit_max_tokens = request.max_tokens if request.max_tokens and request.max_tokens > 0 else None
    logger.info(
        "Max tokens: explicit=%s, expansion=%s, cap=%d",
        explicit_max_tokens,
        settings.enable_token_expansion,
        settings.max_output_tokens,
    )

    messages = [openai_message_to_langchain(m) for m in request.messages]

    allowed_names = settings.allowed_tool_set
    filtered_tools = filter_tools(request.tools, allowed_names)
    logger.info(
        "Tools: incoming=%d, allowed=%d, filtered=%d",
        len(request.tools or []),
        len(allowed_names),
        len(filtered_tools),
    )

    curated_tools = get_tools(allowed_names)
    tools_to_pass = curated_tools if request.tools else []

    model = NemetronChatModel(
        model_name=request.model,
        temperature=request.temperature,
        max_tokens=explicit_max_tokens,
    )

    try:
        # ---- PASS-THROUGH MODE: raw OpenAI-compatible tool_calls response ----
        if tool_mode == "passthrough":
            ai_response = await model.ainvoke(
                messages,
                tools=tools_to_pass,
                temperature=request.temperature,
                max_tokens=explicit_max_tokens,
            )
            return build_tool_calls_response(
                ai_response,
                model=request.model,
                prompt_tokens=sum(len(str(m.content)) for m in messages) // 4,
                completion_tokens=len(str(ai_response.content or "")) // 4,
            )

        # ---- AGENT MODE (default): execute tools internally ----
        agent = NemetronAgent(model=model, tools=curated_tools)

        if request.stream:
            final_content = await agent.arun(
                messages,
                temperature=request.temperature,
                max_tokens=explicit_max_tokens,
                tools=tools_to_pass,
            )
            return StreamingResponse(
                stream_openai_response(final_content, request.model),
                media_type="text/event-stream",
            )

        final_content = await agent.arun(
            messages,
            temperature=request.temperature,
            max_tokens=explicit_max_tokens,
            tools=tools_to_pass,
        )

        return build_openai_response(
            content=final_content,
            model=request.model,
            prompt_tokens=sum(len(str(m.content)) for m in messages) // 4,
            completion_tokens=len(final_content) // 4,
        )
    except Exception as e:
        logger.exception("Error during chat completion")
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.nemetron_model}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=settings.proxy_host, port=settings.proxy_port, reload=False)
