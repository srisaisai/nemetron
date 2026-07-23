from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx

from config import settings


class NemetronClient:
    """Async HTTP client for the local OpenAI-compatible Nemetron API."""

    def __init__(
        self,
        base_url: str = settings.nemetron_base_url,
        api_key: str = settings.nemetron_api_key,
        model: str = settings.nemetron_model,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}{path}"

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        stop: list[str] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        # Merge any extra params from the caller
        for key, value in extra.items():
            if value is not None and key not in payload:
                payload[key] = value

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            response = await client.post(
                self._endpoint("/v1/chat/completions"),
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    async def stream_chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        **extra: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        for key, value in extra.items():
            if value is not None and key not in payload:
                payload[key] = value

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream(
                "POST",
                self._endpoint("/v1/chat/completions"),
                headers=self.headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        data = line[len("data: ") :]
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            response = await client.get(
                self._endpoint("/v1/models"),
                headers=self.headers,
            )
            response.raise_for_status()
            data = response.json()
            return [m.get("id", "") for m in data.get("data", [])]
