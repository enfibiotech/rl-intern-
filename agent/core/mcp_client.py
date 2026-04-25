"""
mcp_client.py — MCP (Model Context Protocol) server integration for RL Intern.

Supports two transports:
  • stdio  — spawns a local process (e.g. npx @modelcontextprotocol/server-filesystem)
  • http   — connects to a remote MCP server over HTTP/SSE

Environment variable substitution is handled automatically:
  "${MY_TOKEN}" in header values → os.environ["MY_TOKEN"]

Usage
-----
    client = MCPClient(servers_config)
    await client.connect_all()

    tools  = client.list_tools()          # → list[ToolSpec]
    result = await client.call_tool("read_file", {"path": "..."})

    await client.disconnect_all()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _substitute_env(value: str) -> str:
    """Replace ${VAR_NAME} with the corresponding environment variable."""
    def replacer(m: re.Match) -> str:
        var = m.group(1)
        result = os.environ.get(var, "")
        if not result:
            log.warning("MCP: env var %s is not set", var)
        return result

    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _resolve_headers(headers: dict[str, str]) -> dict[str, str]:
    return {k: _substitute_env(v) for k, v in headers.items()}


# ---------------------------------------------------------------------------
# Low-level JSON-RPC helpers
# ---------------------------------------------------------------------------

def _jsonrpc(method: str, params: Any = None, id_: int = 1) -> dict:
    msg: dict = {"jsonrpc": "2.0", "method": method, "id": id_}
    if params is not None:
        msg["params"] = params
    return msg


def _jsonrpc_notify(method: str, params: Any = None) -> dict:
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# ---------------------------------------------------------------------------
# Stdio transport
# ---------------------------------------------------------------------------

class StdioMCPConnection:
    """
    Runs an MCP server as a subprocess and communicates over stdin/stdout
    using newline-delimited JSON-RPC.
    """

    def __init__(self, name: str, command: str, args: list[str], env: dict[str, str] | None = None) -> None:
        self.name = name
        self.command = command
        self.args = args
        self.env: dict[str, str] = {**os.environ, **(env or {})}
        self._proc: asyncio.subprocess.Process | None = None
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.env,
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        # Initialize handshake
        await self._send(_jsonrpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "rl-intern", "version": "0.1.0"},
        }))
        await self._send(_jsonrpc_notify("notifications/initialized"))

    async def _read_loop(self) -> None:
        assert self._proc and self._proc.stdout
        while True:
            try:
                line = await self._proc.stdout.readline()
                if not line:
                    break
                msg = json.loads(line.decode())
                msg_id = msg.get("id")
                if msg_id is not None and msg_id in self._pending:
                    fut = self._pending.pop(msg_id)
                    if not fut.done():
                        if "error" in msg:
                            fut.set_exception(RuntimeError(msg["error"].get("message", "MCP error")))
                        else:
                            fut.set_result(msg.get("result"))
            except (json.JSONDecodeError, asyncio.CancelledError):
                break
            except Exception as exc:
                log.debug("MCP stdio read error: %s", exc)
                break

    async def _send(self, msg: dict, expect_reply: bool = True) -> Any:
        assert self._proc and self._proc.stdin
        data = json.dumps(msg).encode() + b"\n"
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()
        if not expect_reply or "id" not in msg:
            return None
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[msg["id"]] = fut
        return await asyncio.wait_for(fut, timeout=30)

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def list_tools(self) -> list[dict]:
        result = await self._send(_jsonrpc("tools/list", id_=self._next_id()))
        return result.get("tools", []) if result else []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = await self._send(_jsonrpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            id_=self._next_id(),
        ))
        if result is None:
            return "No response from MCP server."
        content = result.get("content", [])
        parts = []
        for item in content:
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts) if parts else json.dumps(result)

    async def disconnect(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            try:
                self._proc.terminate()
                await asyncio.wait_for(self._proc.wait(), timeout=3)
            except Exception:
                self._proc.kill()


# ---------------------------------------------------------------------------
# HTTP/SSE transport
# ---------------------------------------------------------------------------

class HttpMCPConnection:
    """
    Connects to a remote MCP server over HTTP.
    Supports both plain JSON-RPC POST and SSE event streams.
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        transport: str = "http",
    ) -> None:
        self.name = name
        self.url = url.rstrip("/")
        self.headers = _resolve_headers(headers or {})
        self.transport = transport  # "http" or "sse"
        self._session = None
        self._id = 0

    async def connect(self) -> None:
        try:
            import aiohttp  # type: ignore
            self._session = aiohttp.ClientSession(headers=self.headers)
            # Verify connectivity with initialize
            await self._rpc("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "rl-intern", "version": "0.1.0"},
            })
        except Exception as exc:
            log.warning("MCP HTTP connect failed for %s: %s", self.name, exc)

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _rpc(self, method: str, params: Any = None) -> Any:
        if self._session is None:
            raise RuntimeError(f"MCP server '{self.name}' not connected.")
        payload = _jsonrpc(method, params, id_=self._next_id())
        try:
            async with self._session.post(
                self.url,
                json=payload,
                timeout=__import__("aiohttp").ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if "error" in data:
                    raise RuntimeError(data["error"].get("message", "MCP error"))
                return data.get("result")
        except Exception as exc:
            log.warning("MCP HTTP RPC error (%s.%s): %s", self.name, method, exc)
            return None

    async def list_tools(self) -> list[dict]:
        result = await self._rpc("tools/list")
        return result.get("tools", []) if result else []

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        result = await self._rpc("tools/call", {"name": tool_name, "arguments": arguments})
        if result is None:
            return "No response from MCP server."
        content = result.get("content", [])
        parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        return "\n".join(parts) if parts else json.dumps(result)

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()


# ---------------------------------------------------------------------------
# MCPClient — manages all connections
# ---------------------------------------------------------------------------

_Connection = StdioMCPConnection | HttpMCPConnection


@dataclass
class MCPToolEntry:
    """Maps a prefixed tool name back to its server connection + original name."""
    prefixed_name: str       # e.g. "filesystem__read_file"
    original_name: str       # e.g. "read_file"
    server_name: str
    connection: _Connection
    schema: dict             # raw MCP tool schema


class MCPClient:
    """
    Central manager that:
      1. Reads the mcpServers dict from the config.
      2. Connects to each server.
      3. Discovers all tools (prefixed to avoid name collisions).
      4. Routes tool calls to the right server.
    """

    def __init__(self, servers_config: dict[str, dict]) -> None:
        self._config = servers_config
        self._connections: dict[str, _Connection] = {}
        self._tool_entries: dict[str, MCPToolEntry] = {}

    # ------------------------------------------------------------------

    async def connect_all(self) -> None:
        for name, cfg in self._config.items():
            transport = cfg.get("transport", "stdio").lower()
            try:
                conn = self._build_connection(name, cfg, transport)
                await conn.connect()
                self._connections[name] = conn
                await self._register_tools(name, conn)
                log.info("MCP: connected to '%s' (%s)", name, transport)
            except Exception as exc:
                log.warning("MCP: failed to connect to '%s': %s", name, exc)

    def _build_connection(self, name: str, cfg: dict, transport: str) -> _Connection:
        if transport == "stdio":
            command_parts = cfg.get("command", "").split()
            command = command_parts[0] if command_parts else "npx"
            args = command_parts[1:] + cfg.get("args", [])
            env_overrides = {k: _substitute_env(v) for k, v in cfg.get("env", {}).items()}
            return StdioMCPConnection(name, command, args, env_overrides)

        elif transport in ("http", "sse"):
            url = cfg.get("url", "")
            headers = cfg.get("headers", {})
            return HttpMCPConnection(name, url, headers, transport)

        else:
            raise ValueError(f"Unknown MCP transport: {transport}")

    async def _register_tools(self, server_name: str, conn: _Connection) -> None:
        tools = await conn.list_tools()
        for tool in tools:
            original = tool.get("name", "")
            # Prefix with server name to avoid collisions: "filesystem__read_file"
            prefixed = f"{server_name}__{original}"
            entry = MCPToolEntry(
                prefixed_name=prefixed,
                original_name=original,
                server_name=server_name,
                connection=conn,
                schema=tool,
            )
            self._tool_entries[prefixed] = entry

    # ------------------------------------------------------------------

    def list_tool_entries(self) -> list[MCPToolEntry]:
        return list(self._tool_entries.values())

    def has_tool(self, prefixed_name: str) -> bool:
        return prefixed_name in self._tool_entries

    async def call_tool(self, prefixed_name: str, arguments: dict) -> str:
        entry = self._tool_entries.get(prefixed_name)
        if entry is None:
            return f"MCP tool not found: {prefixed_name}"
        try:
            return await entry.connection.call_tool(entry.original_name, arguments)
        except Exception as exc:
            return f"MCP tool error ({prefixed_name}): {exc}"

    async def disconnect_all(self) -> None:
        for conn in self._connections.values():
            try:
                await conn.disconnect()
            except Exception:
                pass
        self._connections.clear()
        self._tool_entries.clear()

    # ------------------------------------------------------------------
    # Convert MCP tool schemas → litellm-compatible dicts
    # ------------------------------------------------------------------

    def to_litellm_tools(self) -> list[dict]:
        result = []
        for entry in self._tool_entries.values():
            schema = entry.schema
            # MCP schema uses "inputSchema"; litellm uses "parameters"
            parameters = schema.get("inputSchema") or {
                "type": "object",
                "properties": {},
                "required": [],
            }
            result.append({
                "type": "function",
                "function": {
                    "name": entry.prefixed_name,
                    "description": (
                        f"[MCP:{entry.server_name}] "
                        + schema.get("description", "No description provided.")
                    ),
                    "parameters": parameters,
                },
            })
        return result
