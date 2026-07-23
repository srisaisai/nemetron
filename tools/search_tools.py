from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SearchCodebaseInput(BaseModel):
    path: str = Field(
        description="Absolute path of the directory to search."
    )
    pattern: str = Field(description="Regex pattern to search for.")
    file_pattern: Optional[str] = Field(
        default=None, description="Optional glob pattern to filter files, e.g. '*.py'."
    )
    max_results: int = Field(
        default=200, description="Maximum number of matches to return."
    )


class SearchCodebaseTool(BaseTool):
    name: ClassVar[str] = "search_codebase"
    description: str = (
        "Search file contents in a directory using a regex pattern. Use this when the user "
        "asks you to find, grep, or search for code, functions, variables, or patterns. "
        "Optionally filter by file glob (e.g. '*.py') and limit results."
    )
    args_schema: type[BaseModel] = SearchCodebaseInput

    def _run(
        self,
        path: str,
        pattern: str,
        file_pattern: Optional[str] = None,
        max_results: int = 200,
    ) -> str:
        try:
            dir_path = Path(path).resolve()
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"

            try:
                compiled = re.compile(pattern)
            except re.error as e:
                return f"Error: Invalid regex pattern: {e}"

            matches = []
            files = (
                dir_path.rglob(file_pattern)
                if file_pattern
                else dir_path.rglob("*")
            )

            for file_path in sorted(files):
                if not file_path.is_file():
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f, start=1):
                            if compiled.search(line):
                                rel = file_path.relative_to(dir_path)
                                matches.append(f"{rel}:{line_no}: {line.rstrip()}")
                                if len(matches) >= max_results:
                                    break
                            if len(matches) >= max_results:
                                break
                except Exception:
                    continue
                if len(matches) >= max_results:
                    break

            if not matches:
                return "No matches found."
            return "\n".join(matches)
        except Exception as e:
            return f"Error searching codebase: {e}"
