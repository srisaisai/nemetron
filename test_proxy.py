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

        MockNemetronHandler.call_count += 1

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


if __name__ == "__main__":
    test_proxy_with_mock_upstream()
    test_streaming_response()
    test_passthrough_mode()
    print("All tests passed!")
