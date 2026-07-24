from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from config import settings
from message_utils import message_to_openai, openai_choice_to_ai_message
from nemetron_client import NemetronClient
from stream_bridge import async_iter_to_sync, run_sync

logger = logging.getLogger(__name__)


class NemetronChatModel(BaseChatModel):
    """LangChain ChatModel wrapper around the local Nemetron API."""

    model_name: str = Field(default_factory=lambda: settings.nemetron_model, alias="model")
    nemetron_base_url: str = Field(default_factory=lambda: settings.nemetron_base_url)
    nemetron_api_key: str = Field(default_factory=lambda: settings.nemetron_api_key)
    temperature: Optional[float] = Field(default=0.7)
    max_tokens: Optional[int] = Field(default=None)
    client: Optional[NemetronClient] = Field(default=None, exclude=True)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = NemetronClient(
            base_url=self.nemetron_base_url,
            api_key=self.nemetron_api_key,
            model=self.model_name,
        )

    @property
    def _llm_type(self) -> str:
        return "nemetron"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.nemetron_base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def _convert_tools(tools: List[Any] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        openai_tools = []
        for tool in tools:
            if hasattr(tool, "get_openai_tool"):
                openai_tools.append(tool.get_openai_tool())
            elif hasattr(tool, "args_schema"):
                schema = tool.args_schema.model_json_schema()
                openai_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": getattr(tool, "name", "unknown"),
                            "description": getattr(tool, "description", ""),
                            "parameters": schema,
                        },
                    }
                )
            elif isinstance(tool, dict):
                openai_tools.append(tool)
        return openai_tools or None

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        openai_messages = [message_to_openai(m) for m in messages]
        openai_tools = self._convert_tools(kwargs.get("tools"))

        requested_max = kwargs.get("max_tokens") or self.max_tokens
        initial_max = requested_max or settings.initial_max_tokens
        max_cap = settings.max_output_tokens
        # Only auto-expand when the client did NOT explicitly request a token limit
        enable_expansion = (
            kwargs.get("enable_token_expansion", settings.enable_token_expansion)
            and requested_max is None
        )

        max_tokens = min(initial_max, max_cap)

        while True:
            response = await self.client.chat_completion(
                messages=openai_messages,
                tools=openai_tools,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=max_tokens,
                stop=stop,
            )

            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")

            if finish_reason != "length" or not enable_expansion or max_tokens >= max_cap:
                break

            previous_max = max_tokens
            max_tokens = min(max_tokens * 2, max_cap)
            logger.info(
                "Response truncated at %d tokens; expanding to %d",
                previous_max,
                max_tokens,
            )

        ai_message = openai_choice_to_ai_message(choice)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return run_sync(self._agenerate(messages, stop, run_manager, **kwargs))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        openai_messages = [message_to_openai(m) for m in messages]
        openai_tools = self._convert_tools(kwargs.get("tools"))

        async for chunk in self.client.stream_chat_completion(
            messages=openai_messages,
            tools=openai_tools,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stop=stop,
        ):
            choices = chunk.get("choices") or [{}]
            choice = choices[0] if choices else {}
            delta = choice.get("delta", {}) or {}
            delta_content = delta.get("content") or ""

            # Parse tool-call deltas so passthrough streaming can forward them
            # as proper OpenAI streaming tool_calls (instead of dropping them).
            tool_call_chunks: list[dict[str, Any]] = []
            for tc in delta.get("tool_calls") or []:
                func = tc.get("function", {}) or {}
                tool_call_chunks.append(
                    {
                        "index": tc.get("index", 0),
                        "id": tc.get("id"),
                        "name": func.get("name"),
                        "args": func.get("arguments", "") or "",
                    }
                )

            finish_reason = choice.get("finish_reason")

            if delta_content or tool_call_chunks:
                # AIMessageChunk.tool_call_chunks must be a list (not None), so
                # only supply it when there are actual fragments.
                msg_kwargs: dict[str, Any] = {"content": delta_content}
                if tool_call_chunks:
                    msg_kwargs["tool_call_chunks"] = tool_call_chunks
                yield ChatGenerationChunk(
                    message=AIMessageChunk(**msg_kwargs),
                    generation_info={"finish_reason": finish_reason},
                )

        # Guarantee a terminal chunk so downstream consumers see a stop signal
        # even if the upstream omitted an explicit finish_reason chunk.
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=""),
            generation_info={"finish_reason": "stop"},
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        return async_iter_to_sync(self._astream(messages, stop, run_manager, **kwargs))
