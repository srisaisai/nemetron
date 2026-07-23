from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ListDirectoryInput(BaseModel):
    path: str = Field(description="Absolute path of the directory to list.")
    recursive: bool = Field(default=False, description="List recursively.")


class ListDirectoryTool(BaseTool):
    name: ClassVar[str] = "list_directory"
    description: str = (
        "List files and directories. Use this when the user asks you to explore, "
        "browse, or see what's in a folder. Set recursive=True for deep listing."
    )
    args_schema: type[BaseModel] = ListDirectoryInput

    def _run(self, path: str, recursive: bool = False) -> str:
        try:
            dir_path = Path(path).resolve()
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            lines = [f"Contents of {path}"]
            if recursive:
                for item in sorted(dir_path.rglob("*")):
                    rel = item.relative_to(dir_path)
                    marker = "[D]" if item.is_dir() else "[F]"
                    lines.append(f"{marker} {rel}")
            else:
                for item in sorted(dir_path.iterdir()):
                    marker = "[D]" if item.is_dir() else "[F]"
                    lines.append(f"{marker} {item.name}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing directory: {e}"


class CreateFolderInput(BaseModel):
    path: str = Field(description="Absolute path of the folder to create.")


class CreateFolderTool(BaseTool):
    name: ClassVar[str] = "create_folder"
    description: str = (
        "Create a directory (and parent directories if needed). Use this when the user "
        "asks you to create a folder, make a directory, or set up a project structure."
    )
    args_schema: type[BaseModel] = CreateFolderInput

    def _run(self, path: str) -> str:
        try:
            dir_path = Path(path).resolve()
            dir_path.mkdir(parents=True, exist_ok=True)
            return f"Successfully created folder: {path}"
        except Exception as e:
            return f"Error creating folder: {e}"
