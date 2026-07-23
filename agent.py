from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from chat_model import NemetronChatModel
from config import settings
from text_cleaner import strip_thinking_tags

logger = logging.getLogger(__name__)


class NemetronAgent:
    """LangChain-based agent that manages the model + tool loop."""

    def __init__(
        self,
        model: NemetronChatModel | None = None,
        tools: list[BaseTool] | None = None,
        max_iterations: int = 20,
    ):
        self.model = model or NemetronChatModel()
        self.tools = tools or []
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.max_iterations = max_iterations

    def _ensure_system_prompt(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Prepend a default system prompt if none is present."""
        if not messages:
            return [SystemMessage(content=settings.default_system_prompt)]
        if isinstance(messages[0], SystemMessage):
            return messages
        return [SystemMessage(content=settings.default_system_prompt)] + messages

    def _force_final_prompt(self, history: list[BaseMessage]) -> list[BaseMessage]:
        """Add a system message telling the model to stop calling tools and answer."""
        force_msg = SystemMessage(
            content=(
                "You have already used tools and received their results. "
                "Do NOT call any more tools. Based on all the information gathered, "
                "provide your complete final answer to the user now."
            )
        )
        return self._ensure_system_prompt(list(history)) + [force_msg]

    @staticmethod
    def _tool_call_signature(tool_call: dict) -> str:
        """Create a hashable signature for a tool call to detect duplicates."""
        name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        try:
            args_str = json.dumps(args, sort_keys=True)
        except (TypeError, ValueError):
            args_str = str(args)
        return f"{name}:{args_str}"

    async def _execute_tool_calls(
        self,
        response: AIMessage,
        history: list[BaseMessage],
        seen_signatures: set[str],
        iteration: int,
    ) -> tuple[bool, set[str], list[str]]:
        """Execute tool calls from a response.

        Returns (should_break, updated_seen, completed_tool_names).
        """
        should_break = False
        completed_names: list[str] = []

        for tool_call in response.tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{iteration}")

            sig = self._tool_call_signature(tool_call)
            if sig in seen_signatures:
                logger.warning("Duplicate tool call detected: %s — forcing final answer", sig)
                should_break = True
                break
            seen_signatures.add(sig)

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
            completed_names.append(tool_name)

        return should_break, seen_signatures, completed_names

    @staticmethod
    def _build_step_nudge(completed_names: list[str], iteration: int, is_last: bool) -> HumanMessage:
        """Build a nudge message that guides the model after tool execution."""
        tools_str = ", ".join(completed_names) if completed_names else "tools"
        if is_last:
            content = (
                f"Step {iteration + 1} completed ({tools_str}). "
                "This was your last allowed step. You now MUST provide your final answer "
                "to the user. Do NOT call any more tools. Summarize everything you did "
                "and present the final result."
            )
        else:
            content = (
                f"Step {iteration + 1} completed ({tools_str}). "
                "If you have finished all the work the user asked for, provide your final "
                "answer now WITHOUT calling any tools. "
                "If there are more steps to complete, continue with the next step "
                "(one action at a time)."
            )
        return HumanMessage(content=content)

    async def arun(
        self,
        messages: list[BaseMessage],
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list | None = None,
    ) -> str:
        """Run the agent loop and return the final text response."""
        active_tools = tools if tools is not None else self.tools
        history = self._ensure_system_prompt(list(messages))
        seen_signatures: set[str] = set()

        for iteration in range(self.max_iterations):
            logger.debug("Agent iteration %d/%d", iteration + 1, self.max_iterations)
            is_last = iteration == self.max_iterations - 1

            # On the last iteration, call the model WITHOUT tools to force a text answer
            tools_for_call = [] if is_last else active_tools
            history_for_call = self._force_final_prompt(history) if is_last else history

            response = await self.model.ainvoke(
                history_for_call,
                tools=tools_for_call,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response.content))

            if not response.tool_calls:
                final_content = strip_thinking_tags(response.content or "")
                if not final_content.strip():
                    final_content = (
                        "I received your request but the model returned an empty response. "
                        "Please try again."
                    )
                return final_content

            history.append(response)
            should_break, seen_signatures, completed_names = await self._execute_tool_calls(
                response, history, seen_signatures, iteration
            )

            # Add a step nudge to guide the model toward incremental progress
            history.append(self._build_step_nudge(completed_names, iteration, is_last))

            if should_break:
                # Duplicate detected — force a final answer on next call
                logger.info("Breaking loop due to duplicate tool call")
                forced = await self.model.ainvoke(
                    self._force_final_prompt(history),
                    tools=[],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                final_content = strip_thinking_tags(forced.content or "")
                if not final_content.strip():
                    final_content = (
                        "I was unable to complete the task within the tool loop. "
                        "Please try rephrasing your request."
                    )
                return final_content

        return (
            "I was unable to complete the task within the maximum number of steps. "
            "Please try rephrasing your request or breaking it into smaller steps."
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
        history = self._ensure_system_prompt(list(messages))
        seen_signatures: set[str] = set()

        for iteration in range(self.max_iterations):
            is_last = iteration == self.max_iterations - 1
            tools_for_call = [] if is_last else active_tools
            history_for_call = self._force_final_prompt(history) if is_last else history

            response = await self.model.ainvoke(
                history_for_call,
                tools=tools_for_call,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if not isinstance(response, AIMessage):
                response = AIMessage(content=str(response.content))

            if not response.tool_calls:
                buffered_content = ""
                async for chunk in self.model.astream(
                    history_for_call,
                    tools=tools_for_call,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    buffered_content += chunk.message.content or ""
                cleaned = strip_thinking_tags(buffered_content)
                if not cleaned.strip():
                    cleaned = (
                        "I received your request but the model returned an empty response. "
                        "Please try again."
                    )
                yield AIMessage(content=cleaned)
                return

            history.append(response)
            should_break, seen_signatures, completed_names = await self._execute_tool_calls(
                response, history, seen_signatures, iteration
            )

            # Add a step nudge to guide the model toward incremental progress
            history.append(self._build_step_nudge(completed_names, iteration, is_last))

            if should_break:
                forced = await self.model.ainvoke(
                    self._force_final_prompt(history),
                    tools=[],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                cleaned = strip_thinking_tags(forced.content or "")
                if not cleaned.strip():
                    cleaned = (
                        "I was unable to complete the task within the tool loop. "
                        "Please try rephrasing your request."
                    )
                yield AIMessage(content=cleaned)
                return

        yield AIMessage(
            content=(
                "I was unable to complete the task within the maximum number of steps. "
                "Please try rephrasing your request or breaking it into smaller steps."
            )
        )
