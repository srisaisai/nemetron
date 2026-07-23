from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from chat_model import NemetronChatModel
from config import settings

logger = logging.getLogger(__name__)


class NemetronAgent:
    """LangChain-based agent that manages the model + tool loop."""

    def __init__(
        self,
        model: NemetronChatModel | None = None,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 10,
    ):
        self.model = model or NemetronChatModel()
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.max_iterations = max_iterations

    async def arun(
        self,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list | None = None,
    ) -> str:
        """Run the agent loop and return the final text response."""
        active_tools = tools if tools is not None else self.tools
        history = list(messages)

        for iteration in range(self.max_iterations):
            logger.debug("Agent iteration %d", iteration + 1)
            response = await self.model.ainvoke(
                history,
                tools=active_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response.content))

            if not response.tool_calls:
                return response.content or ""

            history.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{iteration}")

                logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_args))

                if tool_name not in self.tool_map:
                    result_text = f"Error: Tool '{tool_name}' is not available."
                else:
                    try:
                        tool = self.tool_map[tool_name]
                        result = await tool.ainvoke(tool_args)
                        result_text = str(result)
                    except Exception as e:
                        logger.exception("Tool execution failed")
                        result_text = f"Error executing tool {tool_name}: {e}"

                history.append(ToolMessage(content=result_text, tool_call_id=tool_id))

        # If we exhausted iterations, return the last assistant message or a fallback
        return (
            "Reached the maximum number of tool iterations without a final answer."
        )

    async def astream(
        self,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list | None = None,
    ):
        """Run the agent loop and stream the final assistant response."""
        active_tools = tools if tools is not None else self.tools
        history = list(messages)

        for iteration in range(self.max_iterations):
            response = await self.model.ainvoke(
                history,
                tools=active_tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response.content))

            if not response.tool_calls:
                # Stream the final response
                async for chunk in self.model.astream(
                    history,
                    tools=active_tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk
                return

            history.append(response)

            for tool_call in response.tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("args", {})
                tool_id = tool_call.get("id", f"call_{iteration}")

                if tool_name not in self.tool_map:
                    result_text = f"Error: Tool '{tool_name}' is not available."
                else:
                    try:
                        tool = self.tool_map[tool_name]
                        result = await tool.ainvoke(tool_args)
                        result_text = str(result)
                    except Exception as e:
                        result_text = f"Error executing tool {tool_name}: {e}"

                history.append(ToolMessage(content=result_text, tool_call_id=tool_id))

        # Fallback if iterations exhausted
        yield AIMessage(
            content="Reached the maximum number of tool iterations without a final answer."
        )
