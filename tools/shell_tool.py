from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import ClassVar, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from process_manager import process_manager


class RunCommandsInput(BaseModel):
    command: str = Field(description="Shell command to execute.")
    timeout: Optional[int] = Field(
        default=60, description="Timeout in seconds for foreground commands."
    )
    working_dir: Optional[str] = Field(
        default=None, description="Optional working directory for the command."
    )
    background: bool = Field(
        default=False,
        description=(
            "If True, run the command in the background (non-blocking). "
            "Use this for long-running processes like Flask servers, "
            "dev servers, watchers, etc."
        ),
    )


class RunCommandsTool(BaseTool):
    name: ClassVar[str] = "run_commands"
    description: str = (
        "Run a shell command and return stdout/stderr. Use this when the user asks you "
        "to run, execute, test, build, or install something. "
        "For long-running processes (Flask servers, dev servers, watchers), set background=True "
        "so the command runs in the background and returns immediately."
    )
    args_schema: type[BaseModel] = RunCommandsInput

    def _run(
        self,
        command: str,
        timeout: Optional[int] = 60,
        working_dir: Optional[str] = None,
        background: bool = False,
    ) -> str:
        try:
            if background:
                return self._run_background(command, working_dir)
            return self._run_foreground(command, timeout, working_dir)
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error running command: {e}"

    def _run_foreground(self, command, timeout, working_dir) -> str:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout or 60, cwd=working_dir,
        )
        output = []
        if result.stdout:
            output.append("STDOUT:")
            output.append(result.stdout)
        if result.stderr:
            output.append("STDERR:")
            output.append(result.stderr)
        output.append(f"Return code: {result.returncode}")
        return "\n".join(output)

    def _run_background(self, command, working_dir) -> str:
        log_file = Path(tempfile.gettempdir()) / f"nemetron_bg_{int(time.time())}.log"
        process = subprocess.Popen(
            command, shell=True,
            stdout=open(log_file, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            cwd=working_dir, text=True,
        )
        process_manager.add(process.pid, process, str(log_file), command)

        time.sleep(2)
        if process.poll() is not None:
            try:
                output = log_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                output = "(no output)"
            process_manager.remove(process.pid)
            return f"Background process exited immediately (code {process.returncode}).\nOutput:\n{output}"

        initial_output = ""
        try:
            initial_output = log_file.read_text(encoding="utf-8", errors="replace")[:2000]
        except Exception:
            pass

        result = (
            f"Background process started successfully.\n"
            f"PID: {process.pid}\n"
            f"Command: {command}\n"
            f"Log file: {log_file}\n"
        )
        if initial_output.strip():
            result += f"Initial output:\n{initial_output}"
        else:
            result += "No output yet (server may still be starting)."
        return result

