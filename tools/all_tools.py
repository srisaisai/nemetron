from __future__ import annotations

from langchain_core.tools import BaseTool

from tools.file_dir_tools import CreateFolderTool, ListDirectoryTool
from tools.file_read_tools import ReadFileTool, ReadMultipleFilesTool
from tools.file_write_tools import EditFileTool, WriteFileTool
from tools.search_tools import SearchCodebaseTool
from tools.shell_tool import RunCommandsTool
from tools.web_tool import FetchWebContentTool


_ALL_TOOLS: list[BaseTool] = [
    ReadFileTool(),
    ReadMultipleFilesTool(),
    WriteFileTool(),
    EditFileTool(),
    ListDirectoryTool(),
    SearchCodebaseTool(),
    RunCommandsTool(),
    CreateFolderTool(),
    FetchWebContentTool(),
]


def get_tools(allowed_names: set[str] | None = None) -> list[BaseTool]:
    if not allowed_names:
        return list(_ALL_TOOLS)
    return [tool for tool in _ALL_TOOLS if tool.name in allowed_names]


def get_tool_map(allowed_names: set[str] | None = None) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in get_tools(allowed_names)}
