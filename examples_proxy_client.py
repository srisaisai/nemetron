"""Examples: call the RUNNING proxy server (python server.py) as a client."""

from __future__ import annotations

import asyncio


async def example_agent_mode():
    """Agent mode (default): tools executed internally, final answer returned."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
    resp = await client.chat.completions.create(
        model="nemetron-30b",
        messages=[{"role": "user", "content": "List the nemetron folder"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {"type": "boolean", "default": False},
                        },
                        "required": ["path"],
                    },
                },
            }
        ],
        max_tokens=8192,
    )
    print("Agent mode final answer:", resp.choices[0].message.content)


async def example_passthrough_mode():
    """Passthrough mode: raw OpenAI-format tool_calls returned, YOU execute."""
    import httpx

    async with httpx.AsyncClient(timeout=600.0) as http:
        resp = await http.post(
            "http://localhost:8000/v1/chat/completions",
            headers={"X-Tool-Mode": "passthrough"},
            json={
                "model": "nemetron-30b",
                "messages": [{"role": "user", "content": "List the nemetron folder"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "description": "List files and directories",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"},
                                    "recursive": {"type": "boolean", "default": False},
                                },
                                "required": ["path"],
                            },
                        },
                    }
                ],
                "max_tokens": 8192,
            },
        )
        choice = resp.json()["choices"][0]
        if choice["finish_reason"] == "tool_calls":
            for tc in choice["message"]["tool_calls"]:
                print("Raw tool call:", tc["function"]["name"], tc["function"]["arguments"])
        else:
            print("Content:", choice["message"]["content"])


if __name__ == "__main__":
    # Requires the proxy server running (python server.py). Uncomment one:
    # asyncio.run(example_agent_mode())
    # asyncio.run(example_passthrough_mode())
    print("Start the server first (python server.py), then uncomment an example.")
