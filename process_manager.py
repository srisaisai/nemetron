from __future__ import annotations

import threading
from typing import Any


class ProcessManager:
    """Tracks background processes started by run_commands(background=True)."""

    def __init__(self):
        self._processes: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(self, pid: int, process, log_file: str, command: str) -> None:
        with self._lock:
            self._processes[pid] = {
                "process": process,
                "log_file": log_file,
                "command": command,
            }

    def get(self, pid: int) -> dict[str, Any] | None:
        with self._lock:
            return self._processes.get(pid)

    def remove(self, pid: int) -> None:
        with self._lock:
            self._processes.pop(pid, None)

    def list_all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "pid": pid,
                    "command": info["command"],
                    "log_file": info["log_file"],
                    "running": info["process"].poll() is None,
                }
                for pid, info in self._processes.items()
            ]

    def cleanup(self) -> None:
        """Stop all tracked background processes."""
        with self._lock:
            for pid, info in self._processes.items():
                try:
                    if info["process"].poll() is None:
                        info["process"].terminate()
                except Exception:
                    pass
            self._processes.clear()


process_manager = ProcessManager()
