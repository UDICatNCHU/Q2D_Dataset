from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MCPClient:
    """Simple MCP client that communicates with the server via stdio."""

    server_script: str
    process: Optional[subprocess.Popen] = field(default=None, init=False)

    def start(self) -> None:
        """Launch the MCP server process."""
        if self.process is not None:
            return

        self.process = subprocess.Popen(
            [sys.executable, self.server_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

        ready_msg = self.process.stdout.readline()
        print(f"🚀 {ready_msg.strip()}")

    def stop(self) -> None:
        """Terminate the MCP server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def __enter__(self) -> "MCPClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - cleanup
        self.stop()

    def call_tool(self, tool: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call an MCP tool with the provided arguments."""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError("Client is not running")

        request = {"tool": tool, "args": args or {}}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        return json.loads(response_line)
