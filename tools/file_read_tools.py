from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ReadFileInput(BaseModel):
    path: str = Field(description="Absolute path of the file to read.")
    start_line: Optional[int] = Field(default=None, description="Optional 1-based start line.")
    end_line: Optional[int] = Field(default=None, description="Optional 1-based end line (inclusive).")


class ReadFileTool(BaseTool):
    name: ClassVar[str] = "read_file"
    description: str = "Read the contents of a file. Optionally specify start_line and end_line to read a range."
    args_schema: type[BaseModel] = ReadFileInput

    def _run(self, path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total = len(lines)
            start = (start_line or 1) - 1
            end = (end_line or total) - 1
            start = max(0, min(start, total - 1))
            end = max(start, min(end, total - 1))

            selected = lines[start : end + 1]
            content = "".join(selected)
            header = f"--- {path} (lines {start + 1}-{end + 1} of {total}) ---\n"
            return header + content
        except Exception as e:
            return f"Error reading file: {e}"


class ReadMultipleFilesInput(BaseModel):
    paths: list[str] = Field(description="List of absolute file paths to read.")


class ReadMultipleFilesTool(BaseTool):
    name: ClassVar[str] = "read_multiple_files"
    description: str = "Read the contents of multiple files at once."
    args_schema: type[BaseModel] = ReadMultipleFilesInput

    def _run(self, paths: list[str]) -> str:
        results = []
        for path in paths:
            single = ReadFileTool()
            results.append(single._run(path))
            results.append("")
        return "\n".join(results)
