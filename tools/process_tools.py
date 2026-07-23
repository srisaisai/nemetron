from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from process_manager import process_manager


class StopProcessInput(BaseModel):
    pid: int = Field(description="Process ID to stop.")


class StopProcessTool(BaseTool):
    name: ClassVar[str] = "stop_process"
    description: str = (
        "Stop a background process by its PID. Use this to stop servers or "
        "long-running processes that were started with run_commands(background=True)."
    )
    args_schema: type[BaseModel] = StopProcessInput

    def _run(self, pid: int) -> str:
        info = process_manager.get(pid)
        if not info:
            return f"Error: No background process found with PID {pid}."
        try:
            info["process"].terminate()
            time.sleep(1)
            if info["process"].poll() is None:
                info["process"].kill()
            process_manager.remove(pid)
            return f"Stopped background process PID {pid} (was: {info['command']})"
        except Exception as e:
            return f"Error stopping process {pid}: {e}"


class ReadProcessOutputInput(BaseModel):
    pid: int = Field(description="Process ID to read output from.")
    lines: int = Field(default=50, description="Number of recent lines to read.")


class ReadProcessOutputTool(BaseTool):
    name: ClassVar[str] = "read_process_output"
    description: str = (
        "Read the recent output of a background process by its PID. "
        "Use this to check on servers or long-running processes started with "
        "run_commands(background=True)."
    )
    args_schema: type[BaseModel] = ReadProcessOutputInput

    def _run(self, pid: int, lines: int = 50) -> str:
        info = process_manager.get(pid)
        if not info:
            return f"Error: No background process found with PID {pid}."
        try:
            log_path = Path(info["log_file"])
            if not log_path.exists():
                return f"No output yet for PID {pid}."
            all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
            status = "running" if info["process"].poll() is None else "exited"
            return f"Process PID {pid} ({status}):\n" + "\n".join(recent)
        except Exception as e:
            return f"Error reading output for PID {pid}: {e}"
