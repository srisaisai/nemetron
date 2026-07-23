from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class WriteFileInput(BaseModel):
    path: str = Field(description="Absolute path where the file should be written.")
    content: str = Field(description="Content to write into the file.")
    overwrite: bool = Field(default=False, description="Whether to overwrite if the file exists.")


class WriteFileTool(BaseTool):
    name: ClassVar[str] = "write_file"
    description: str = (
        "Write content to a file. Use this when the user asks you to create a new file. "
        "Set overwrite=True to replace an existing file."
    )
    args_schema: type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str, overwrite: bool = False) -> str:
        try:
            file_path = Path(path).resolve()
            if file_path.exists() and not overwrite:
                return f"Error: File already exists. Set overwrite=True to replace: {path}"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote file: {path}"
        except Exception as e:
            return f"Error writing file: {e}"


class EditFileInput(BaseModel):
    path: str = Field(description="Absolute path of the file to edit.")
    old_text: str = Field(description="Exact text to replace.")
    new_text: str = Field(description="Replacement text.")


class EditFileTool(BaseTool):
    name: ClassVar[str] = "edit_file"
    description: str = (
        "Replace old_text with new_text in a file. Use this when the user asks you to "
        "modify, update, or fix a specific part of an existing file. The old_text must "
        "match exactly."
    )
    args_schema: type[BaseModel] = EditFileInput

    def _run(self, path: str, old_text: str, new_text: str) -> str:
        try:
            file_path = Path(path).resolve()
            if not file_path.exists():
                return f"Error: File not found: {path}"
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if old_text not in content:
                return f"Error: old_text not found in {path}"
            content = content.replace(old_text, new_text, 1)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully edited file: {path}"
        except Exception as e:
            return f"Error editing file: {e}"
