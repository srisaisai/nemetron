"""End-to-end test using a mock Nemetron upstream server."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastapi.testclient import TestClient

from server import app


class MockNemetronHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible handler for testing."""

    call_count = 0

    def log_message(self, format, *args):
        pass

    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_sse(self, events: list[dict]):
        """Send a list of pre-built SSE data events, then [DONE]."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for evt in events:
            self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [{"id": "nemetron-30b"}]})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = body.get("messages", [])
        tools = body.get("tools", [])
        stream = body.get("stream", False)

        MockNemetronHandler.call_count += 1

        # ---- Streaming responses (for passthrough/agent streaming tests) ----
        if stream:
            self._handle_stream(tools, messages)
            return

        # If tools are provided and last message is not a tool response, return a tool call
        if tools and messages[-1]["role"] == "user":
            self._send_json(
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "nemetron-30b",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_123",
                                        "type": "function",
                                        "function": {
                                            "name": "list_directory",
                                            "arguments": json.dumps({"path": "C:\\\\Users\\\\saiof\\\\nemetron"}),
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                },
            )
            return

        # Otherwise return a final answer
        self._send_json(
            200,
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "nemetron-30b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Done. I listed the directory for you.",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    def _handle_stream(self, tools, messages):
        """Emit an SSE stream.

        * tools + user message  -> stream a tool call split across chunks
          (exercises fragment reassembly by the proxy).
        * otherwise             -> stream text containing a thinking block
          split across chunks (exercises streaming thinking-tag stripping).
        """
        created = int(time.time())

        if tools and messages and messages[-1]["role"] == "user":
            # Tool call fragmented across several deltas.
            events = [
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "nemetron-30b",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_stream1",
                                        "type": "function",
                                        "function": {
                                            "name": "list_directory",
                                            "arguments": "{\"path\": \"",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "nemetron-30b",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": "C:\\\\Users"},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "nemetron-30b",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": "\"}"},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-stream",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "nemetron-30b",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                },
            ]
            self._send_sse(events)
            return

        # Text with a thinking tag split across chunk boundaries.
        # Expected after stripping: "Let me think. Here is my answer."
        events = [
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "nemetron-30b",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "Let me think. "}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "nemetron-30b",
                "choices": [
                    {"index": 0, "delta": {"content": "<thi"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "nemetron-30b",
                "choices": [
                    {"index": 0, "delta": {"content": "nk>secret reasoning"}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "nemetron-30b",
                "choices": [
                    {"index": 0, "delta": {"content": "</think>Here is my answer."}, "finish_reason": None}
                ],
            },
            {
                "id": "chatcmpl-stream",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "nemetron-30b",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"}
                ],
            },
        ]
        self._send_sse(events)


def start_mock_server(port: int = 11435):
    server = HTTPServer(("127.0.0.1", port), MockNemetronHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_proxy_with_mock_upstream():
    mock_port = 11435
    server = start_mock_server(mock_port)
    time.sleep(0.5)

    # Point config to mock server for this test
    import config

    original_base_url = config.settings.nemetron_base_url
    config.settings.nemetron_base_url = f"http://127.0.0.1:{mock_port}"

    try:
        client = TestClient(app)

        # Test models endpoint
        models_resp = client.get("/v1/models")
        assert models_resp.status_code == 200
        print("Models:", models_resp.json())

        # Test chat completion with tool call loop (agent mode)
        MockNemetronHandler.call_count = 0
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nemetron-30b",
                "messages": [{"role": "user", "content": "List the nemetron folder"}],
                "tool_mode": "agent",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "description": "Lists files",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "some_other_tool",
                            "description": "Should be filtered out",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ],
                "max_tokens": 1024,
            },
        )
        print("Chat status:", resp.status_code)
        print("Chat body:", resp.json())

        # The agent should have looped: first call got tool call, second call got final answer
        assert resp.status_code == 200
        assert MockNemetronHandler.call_count >= 2
        print(f"Mock upstream was called {MockNemetronHandler.call_count} times")

    finally:
        config.settings.nemetron_base_url = original_base_url
        server.shutdown()



def test_streaming_response():
    mock_port = 11436
    server = start_mock_server(mock_port)
    time.sleep(0.5)

    import config

    original_base_url = config.settings.nemetron_base_url
    config.settings.nemetron_base_url = f"http://127.0.0.1:{mock_port}"

    try:
        client = TestClient(app)
        MockNemetronHandler.call_count = 0
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nemetron-30b",
                "messages": [{"role": "user", "content": "Say hello"}],
                "tool_mode": "agent",
                "stream": True,
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200
        body = resp.text
        assert "data:" in body
        assert "[DONE]" in body
        print("Streaming response OK")
    finally:
        config.settings.nemetron_base_url = original_base_url
        server.shutdown()


def test_passthrough_mode():
    mock_port = 11437
    server = start_mock_server(mock_port)
    time.sleep(0.5)

    import config

    original_base_url = config.settings.nemetron_base_url
    config.settings.nemetron_base_url = f"http://127.0.0.1:{mock_port}"

    try:
        client = TestClient(app)
        MockNemetronHandler.call_count = 0

        # Test via header
        resp = client.post(
            "/v1/chat/completions",
            headers={"X-Tool-Mode": "passthrough"},
            json={
                "model": "nemetron-30b",
                "messages": [{"role": "user", "content": "List the nemetron folder"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "description": "Lists files",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        print("Passthrough response:", json.dumps(data, indent=2))

        # In passthrough mode the tool_calls should be returned RAW, not executed.
        # The upstream should only be called ONCE (no agent loop).
        assert MockNemetronHandler.call_count == 1, (
            f"Expected 1 upstream call (no agent loop), got {MockNemetronHandler.call_count}"
        )

        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tool_calls = choice["message"]["tool_calls"]
        assert tool_calls[0]["function"]["name"] == "list_directory"
        assert tool_calls[0]["type"] == "function"
        print("Passthrough mode OK: raw tool_calls returned, single upstream call")
    finally:
        config.settings.nemetron_base_url = original_base_url
        server.shutdown()


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE ``data:`` lines (excluding [DONE]) into a list of dicts."""
    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                events.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                continue
    return events


def test_passthrough_streaming_tool_calls():
    """Streaming passthrough must forward tool_calls as proper delta.tool_calls
    (not plain text) so the client can execute them."""
    mock_port = 11438
    server = start_mock_server(mock_port)
    time.sleep(0.5)

    import config

    original_base_url = config.settings.nemetron_base_url
    config.settings.nemetron_base_url = f"http://127.0.0.1:{mock_port}"

    try:
        client = TestClient(app)
        MockNemetronHandler.call_count = 0

        resp = client.post(
            "/v1/chat/completions",
            headers={"X-Tool-Mode": "passthrough"},
            json={
                "model": "nemetron-30b",
                "messages": [{"role": "user", "content": "List the folder"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_directory",
                            "description": "Lists files",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "stream": True,
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200, resp.text
        assert MockNemetronHandler.call_count == 1, "passthrough must not loop"

        events = _parse_sse(resp.text)
        assert events, "expected at least one SSE event"

        # Reassemble streaming tool_calls by index, mirroring an OpenAI client.
        assembled: dict[int, dict] = {}
        finish_reason = None
        for evt in events:
            choice = evt["choices"][0]
            delta = choice.get("delta", {})
            for tc in delta.get("tool_calls", []) or []:
                idx = tc.get("index", 0)
                slot = assembled.setdefault(idx, {"id": None, "name": None, "args": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        assert finish_reason == "tool_calls", f"got {finish_reason!r}"
        assert 0 in assembled, "no tool_calls were streamed to the client"
        slot = assembled[0]
        assert slot["id"] == "call_stream1", slot
        assert slot["name"] == "list_directory", slot
        args = json.loads(slot["args"])
        assert args == {"path": "C:\\Users"}, slot
        print("Streaming passthrough tool_calls OK:", slot)
    finally:
        config.settings.nemetron_base_url = original_base_url
        server.shutdown()


def test_passthrough_streaming_strips_thinking():
    """Streaming passthrough must NOT leak thinking tags to the client."""
    mock_port = 11439
    server = start_mock_server(mock_port)
    time.sleep(0.5)

    import config

    original_base_url = config.settings.nemetron_base_url
    config.settings.nemetron_base_url = f"http://127.0.0.1:{mock_port}"

    try:
        client = TestClient(app)
        MockNemetronHandler.call_count = 0

        resp = client.post(
            "/v1/chat/completions",
            headers={"X-Tool-Mode": "passthrough"},
            json={
                "model": "nemetron-30b",
                "messages": [{"role": "user", "content": "Say hello"}],
                "stream": True,
                "max_tokens": 1024,
            },
        )
        assert resp.status_code == 200, resp.text

        events = _parse_sse(resp.text)
        content = ""
        finish_reason = None
        for evt in events:
            choice = evt["choices"][0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                content += delta["content"]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

        assert finish_reason == "stop", finish_reason
        assert "<" not in content, f"thinking tag leaked into stream: {content!r}"
        assert "secret" not in content, f"thinking content leaked: {content!r}"
        assert content == "Let me think. Here is my answer.", repr(content)
        print("Streaming passthrough thinking-strip OK:", repr(content))
    finally:
        config.settings.nemetron_base_url = original_base_url
        server.shutdown()


def test_stream_thinking_filter_unit():
    """Pure unit tests for StreamThinkingFilter across chunk boundaries."""
    from text_cleaner import StreamThinkingFilter

    OPEN = chr(60) + "think" + chr(62)
    CLOSE = chr(60) + "/think" + chr(62)

    # Complete block in one chunk
    f = StreamThinkingFilter()
    assert f.feed("before " + OPEN + "secret" + CLOSE + " after") == "before  after"
    assert f.flush() == ""

    # Block split across chunk boundaries (open tag fragmented)
    f = StreamThinkingFilter()
    assert f.feed("hi " + OPEN[:4]) == "hi "
    assert f.feed(OPEN[4:] + "secret") == ""
    assert f.feed(CLOSE + "done") == "done"
    assert f.flush() == ""

    # Close tag fragmented across chunks
    f = StreamThinkingFilter()
    assert f.feed(OPEN + "reasoning") == ""
    assert f.feed(CLOSE[:5]) == ""
    assert f.feed(CLOSE[5:] + "tail") == "tail"
    assert f.flush() == ""

    # Unterminated thinking block: discard everything
    f = StreamThinkingFilter()
    assert f.feed("x" + OPEN + "y") == "x"
    assert f.flush() == ""

    # Case-insensitive
    f = StreamThinkingFilter()
    big_open = chr(60) + "THINKING" + chr(62)
    big_close = chr(60) + "/THINKING" + chr(62)
    assert f.feed("a" + big_open + "hidden" + big_close + "b") == "ab"
    assert f.flush() == ""

    # No tags at all
    f = StreamThinkingFilter()
    assert f.feed("just plain text") == "just plain text"
    assert f.flush() == ""

    print("StreamThinkingFilter unit tests OK")


if __name__ == "__main__":
    test_proxy_with_mock_upstream()
    test_streaming_response()
    test_passthrough_mode()
    test_passthrough_streaming_tool_calls()
    test_passthrough_streaming_strips_thinking()
    test_stream_thinking_filter_unit()
    print("All tests passed!")
