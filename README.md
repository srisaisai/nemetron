# Nemetron LangChain Proxy

A lightweight **OpenAI-compatible proxy** for your local **Nemetron 30B** model, built with **FastAPI** and **LangChain Core**.

It sits between VS Code extensions (Cline, Continue, Roo Code, etc.) and your local Nemetron API at `http://localhost:11434`, adding:

- ✅ LangChain-based tool-call orchestration
- ✅ Curated tool set (filters out VS Code's massive tool schema)
- ✅ Higher output token limits (not clamped to 4096)
- ✅ OpenAI-compatible `/v1/chat/completions` endpoint

---

## Project Structure

```
nemetron/
├── config.py                 # Pydantic settings from .env
├── server.py                 # FastAPI OpenAI-compatible proxy
├── chat_model.py             # Custom LangChain BaseChatModel
├── agent.py                  # Tool-call loop orchestrator
├── nemetron_client.py        # HTTP client for localhost:11434
├── message_utils.py          # OpenAI <-> LangChain message conversion
├── schemas.py                # Pydantic request/response models
├── response_builder.py       # OpenAI response formatting
├── stream_bridge.py          # Sync/async bridge utilities
├── tools/                    # Curated tool implementations
│   ├── file_read_tools.py
│   ├── file_write_tools.py
│   ├── file_dir_tools.py
│   ├── search_tools.py
│   ├── shell_tool.py
│   └── web_tool.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
cd nemetron
pip install -r requirements.txt
```

### 2. Configure environment

Copy the example file and adjust if needed:

```bash
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEMETRON_BASE_URL` | `http://10.33.11.12:8103` | Your local Nemetron API |
| `NEMETRON_MODEL` | `nemetron-30b` | Model name reported to clients |
| `PROXY_PORT` | `8000` | Port the proxy listens on |
| `INITIAL_MAX_TOKENS` | `8192` | Starting output token limit |
| `MAX_OUTPUT_TOKENS` | `262000` | Maximum output token ceiling |
| `ENABLE_TOKEN_EXPANSION` | `true` | Auto-expand tokens if truncated |
| `ALLOWED_TOOLS` | comma-separated | Curated tool names exposed to the model |

### 3. Start the proxy

```bash
python server.py
```

The proxy will be available at:

```
http://localhost:8000/v1
```

---

## VS Code Configuration

In your VS Code extension (Cline / Continue / Roo Code), configure the OpenAI-compatible provider:

- **Base URL:** `http://localhost:8000/v1`
- **API Key:** anything (e.g., `not-needed`)
- **Model:** `nemetron-30b`

The extension will now talk to the proxy, which handles tool filtering and orchestration before forwarding to your local Nemetron API.

---

## Available Curated Tools

| `ENABLE_TOKEN_EXPANSION` | `true` | Auto-expand tokens if truncated |
| `ALLOWED_TOOLS` | comma-separated | Curated tool names exposed to the model |

| Tool | Purpose |
|------|---------|
| `read_file` | Read a file or a line range |
| `read_multiple_files` | Read several files at once |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace exact text in a file |
| `list_directory` | List directory contents |
| `search_codebase` | Regex search across files |
| `run_commands` | Execute a shell command |
| `create_folder` | Create directories |
| `fetch_web_content` | Fetch web page text |

Tools sent by VS Code that are **not** in this list are filtered out before reaching the model.

---

## How It Works

```
VS Code Extension
        │
        ▼  OpenAI API
┌──────────────────────┐
│  Nemetron Proxy      │  ← filters tools, fixes token limits
│  (FastAPI +          │  ← LangChain agent loop
│   LangChain Core)    │
└──────────┬───────────┘
           │
           ▼  OpenAI API
┌──────────────────────┐
│  Local Nemetron API  │
│  @ localhost:11434   │
└──────────────────────┘
```

---

## Two Tool Modes

The proxy supports two modes, selected via the `X-Tool-Mode` header or a `tool_mode` field in the request body.

### 1. Agent mode (default) — for VS Code extensions

The proxy runs a LangChain agent loop: it sends tools to the model, executes any tool calls locally, feeds results back, and returns the **final answer only**. This is what Cline / Continue / Roo Code expect.

```
POST /v1/chat/completions        (no header, or X-Tool-Mode: agent)
```

### 2. Passthrough mode — for programmatic / LangChain use

The proxy forwards messages + filtered tools to the model **once** and returns the raw OpenAI-compatible response, including `tool_calls` (with `finish_reason: "tool_calls"`) **without executing them**. You execute the tools yourself and send the follow-up request.

```
POST /v1/chat/completions
X-Tool-Mode: passthrough
```

or in the body:

```json
{ "tool_mode": "passthrough", ... }
```

Example response in passthrough mode:

```json
{
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "id": "call_123",
        "type": "function",
        "function": {"name": "list_directory", "arguments": "{\"path\": \"...\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

---

## Programmatic Usage (LangChain)

You can also import the components directly in your own Python programs, without running the server:

```python
import sys
sys.path.insert(0, r"C:\Users\saiof\nemetron")   # or run from inside the folder

from langchain_core.messages import HumanMessage, ToolMessage
from chat_model import NemetronChatModel
from agent import NemetronAgent
from tools import get_tools

# Option A: direct model + bind_tools (you control tool execution)
model = NemetronChatModel(max_tokens=8192)
tools = get_tools()
llm = model.bind_tools(tools)
resp = llm.invoke([HumanMessage(content="What files are in the current folder?")])
print(resp.tool_calls)   # standard LangChain tool_calls

# Option B: full agent loop (tools executed automatically)
agent = NemetronAgent(model=model, tools=tools)
answer = agent.arun([HumanMessage(content="List the folder and summarize it.")])
```

See `examples.py` (direct LangChain usage) and `examples_proxy_client.py` (calling the running server in both modes).

---

## Testing

### Health check

```bash
curl http://localhost:8000/health
```

### List models

```bash
curl http://localhost:8000/v1/models
```

### Simple chat completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemetron-30b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1024
  }'
```

---

## Notes

- The proxy does **not** use `langchain-community`. Only `langchain-core` is required.
- Streaming responses are supported; the agent loop runs internally first, then the final answer is streamed.
- If the upstream Nemetron API is not running, the proxy will return an error on the first request.

## Thinking Tag Filtering

The proxy automatically strips `<think>...</think>` and similar reasoning tags from model responses before returning them to VS Code. You only see the final answer.

---

## Progressive Token Expansion

Instead of a fixed token limit, the proxy uses progressive expansion:

1. Starts with `INITIAL_MAX_TOKENS` (default 8192)
2. If the model response is truncated (`finish_reason == "length"`), doubles the limit
3. Repeats until complete or `MAX_OUTPUT_TOKENS` (262000) is reached

Set `ENABLE_TOKEN_EXPANSION=false` to disable and use a fixed limit.

---
