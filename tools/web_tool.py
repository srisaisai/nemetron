from __future__ import annotations

from typing import ClassVar

import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class FetchWebContentInput(BaseModel):
    url: str = Field(description="URL to fetch.")
    max_length: int = Field(
        default=50000, description="Maximum characters to return."
    )


class FetchWebContentTool(BaseTool):
    name: ClassVar[str] = "fetch_web_content"
    description: str = (
        "Fetch content from a URL and return it as text. Use this when the user asks you "
        "to read a webpage, fetch documentation, or retrieve online content."
    )
    args_schema: type[BaseModel] = FetchWebContentInput

    def _run(self, url: str, max_length: int = 50000) -> str:
        try:
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()
                text = response.text
                if len(text) > max_length:
                    text = text[:max_length] + "\n... [truncated]"
                return text
        except Exception as e:
            return f"Error fetching URL: {e}"
