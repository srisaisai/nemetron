from __future__ import annotations

import subprocess
from typing import ClassVar, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class RunCommandsInput(BaseModel):
    command: str = Field(description="Shell command to execute.")
    timeout: Optional[int] = Field(
        default=60, description="Timeout in seconds."
    )
    working_dir: Optional[str] = Field(
        default=None, description="Optional working directory for the command."
    )


class RunCommandsTool(BaseTool):
    name: ClassVar[str] = "run_commands"
    description: str = (
        "Run a shell command and return stdout/stderr. Use this when the user asks you "
        "to run, execute, test, build, or install something. You can specify a working "
        "directory and timeout."
    )
    args_schema: type[BaseModel] = RunCommandsInput

    def _run(
        self,
        command: str,
        timeout: Optional[int] = 60,
        working_dir: Optional[str] = None,
    ) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout or 60,
                cwd=working_dir,
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
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error running command: {e}"
