"""Examples: use the Nemetron model programmatically with LangChain.

Run from inside the `nemetron` folder. Nemetron API must be running.
"""

from __future__ import annotations

import asyncio


async def example_simple_chat():
    """Direct LangChain usage with the chat model (simple chat)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from chat_model import NemetronChatModel

    model = NemetronChatModel(max_tokens=8192)
    response = await model.ainvoke(
        [
            SystemMessage(content="You are a helpful coding assistant."),
            HumanMessage(content="What is a Python decorator, in one sentence?"),
        ]
    )
    print("Simple chat:", response.content)


async def example_langchain_tool_calling():
    """LangChain tool calling via bind_tools - YOU control execution."""
    from langchain_core.messages import HumanMessage, ToolMessage

    from chat_model import NemetronChatModel
    from tools import get_tools

    model = NemetronChatModel(max_tokens=8192)
    tools = get_tools()
    model_with_tools = model.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [HumanMessage(content="What files are in the current folder?")]
    response = await model_with_tools.ainvoke(messages)

    if response.tool_calls:
        for tc in response.tool_calls:
            print("Model wants to call:", tc["name"], "with", tc["args"])
            result = await tool_map[tc["name"]].ainvoke(tc["args"])
            messages.append(response)
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            print("Tool result (truncated):", str(result)[:200])
    else:
        print("Final answer:", response.content)


async def example_agent_loop():
    """Full agent loop (tools executed automatically)."""
    from langchain_core.messages import HumanMessage

    from agent import NemetronAgent
    from chat_model import NemetronChatModel
    from tools import get_tools

    agent = NemetronAgent(model=NemetronChatModel(max_tokens=8192), tools=get_tools())
    answer = await agent.arun([HumanMessage(content="List the current folder, then summarize it.")])
    print("Agent answer:", answer)


if __name__ == "__main__":
    # Uncomment what you want to run (direct LangChain usage):
    # asyncio.run(example_simple_chat())
    # asyncio.run(example_langchain_tool_calling())
    # asyncio.run(example_agent_loop())
    print("Uncomment an example in __main__ to run it.")
