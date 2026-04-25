"""
tests/test_mcp.py — unit tests for MCPClient and ToolRouter MCP integration.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.core.mcp_client import MCPClient, HttpMCPConnection, StdioMCPConnection, _substitute_env
from agent.core.tool_router import ToolRouter


# ---------------------------------------------------------------------------
# _substitute_env
# ---------------------------------------------------------------------------

class TestSubstituteEnv:
    def test_replaces_known_var(self, monkeypatch):
        monkeypatch.setenv("MY_TOKEN", "abc123")
        assert _substitute_env("Bearer ${MY_TOKEN}") == "Bearer abc123"

    def test_missing_var_returns_empty(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        result = _substitute_env("${MISSING_VAR}")
        assert result == ""

    def test_no_substitution_needed(self):
        assert _substitute_env("plain string") == "plain string"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("A", "hello")
        monkeypatch.setenv("B", "world")
        assert _substitute_env("${A} ${B}") == "hello world"


# ---------------------------------------------------------------------------
# MCPClient config parsing
# ---------------------------------------------------------------------------

class TestMCPClientConfig:
    def test_builds_stdio_connection(self):
        cfg = {
            "my-server": {
                "transport": "stdio",
                "command": "npx -y @modelcontextprotocol/server-filesystem /tmp",
                "args": [],
            }
        }
        client = MCPClient(cfg)
        conn = client._build_connection("my-server", cfg["my-server"], "stdio")
        assert isinstance(conn, StdioMCPConnection)
        assert conn.name == "my-server"

    def test_builds_http_connection(self):
        cfg = {
            "remote": {
                "transport": "http",
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer tok"},
            }
        }
        client = MCPClient(cfg)
        conn = client._build_connection("remote", cfg["remote"], "http")
        assert isinstance(conn, HttpMCPConnection)
        assert conn.url == "https://example.com/mcp"

    def test_unknown_transport_raises(self):
        cfg = {"bad": {"transport": "ftp", "url": "ftp://x"}}
        client = MCPClient(cfg)
        with pytest.raises(ValueError, match="Unknown MCP transport"):
            client._build_connection("bad", cfg["bad"], "ftp")

    def test_empty_config_connects_nothing(self):
        client = MCPClient({})
        assert client.list_tool_entries() == []


# ---------------------------------------------------------------------------
# MCPClient tool registration (mocked connection)
# ---------------------------------------------------------------------------

class TestMCPToolRegistration:
    @pytest.mark.asyncio
    async def test_registers_tools_from_server(self):
        mock_conn = AsyncMock()
        mock_conn.connect = AsyncMock()
        mock_conn.list_tools = AsyncMock(return_value=[
            {
                "name": "read_file",
                "description": "Read a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "write_file",
                "description": "Write a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"],
                },
            },
        ])

        client = MCPClient({})
        client._connections["filesystem"] = mock_conn
        await client._register_tools("filesystem", mock_conn)

        entries = client.list_tool_entries()
        names = [e.prefixed_name for e in entries]
        assert "filesystem__read_file" in names
        assert "filesystem__write_file" in names

    @pytest.mark.asyncio
    async def test_prefixed_names_avoid_collision(self):
        client = MCPClient({})

        mock_conn_a = AsyncMock()
        mock_conn_a.list_tools = AsyncMock(return_value=[
            {"name": "search", "description": "Search A", "inputSchema": {"type": "object", "properties": {}}}
        ])
        mock_conn_b = AsyncMock()
        mock_conn_b.list_tools = AsyncMock(return_value=[
            {"name": "search", "description": "Search B", "inputSchema": {"type": "object", "properties": {}}}
        ])

        await client._register_tools("server_a", mock_conn_a)
        await client._register_tools("server_b", mock_conn_b)

        names = [e.prefixed_name for e in client.list_tool_entries()]
        assert "server_a__search" in names
        assert "server_b__search" in names

    @pytest.mark.asyncio
    async def test_call_tool_routes_to_correct_server(self):
        client = MCPClient({})

        mock_conn = AsyncMock()
        mock_conn.list_tools = AsyncMock(return_value=[
            {"name": "ping", "description": "Ping", "inputSchema": {"type": "object", "properties": {}}}
        ])
        mock_conn.call_tool = AsyncMock(return_value="pong")

        await client._register_tools("myserver", mock_conn)
        result = await client.call_tool("myserver__ping", {})
        assert result == "pong"
        mock_conn.call_tool.assert_called_once_with("ping", {})

    @pytest.mark.asyncio
    async def test_call_unknown_tool_returns_error(self):
        client = MCPClient({})
        result = await client.call_tool("nonexistent__tool", {})
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# ToolRouter MCP integration
# ---------------------------------------------------------------------------

class TestToolRouterMCP:
    def _make_router_with_mcp(self) -> tuple[ToolRouter, MCPClient]:
        client = MCPClient({})
        router = ToolRouter(mcp_client=client)
        return router, client

    @pytest.mark.asyncio
    async def test_mcp_tools_appear_in_litellm_list(self):
        router, client = self._make_router_with_mcp()

        mock_conn = AsyncMock()
        mock_conn.list_tools = AsyncMock(return_value=[
            {"name": "ls", "description": "List files", "inputSchema": {"type": "object", "properties": {}}}
        ])
        await client._register_tools("fs", mock_conn)

        tool_defs = router.to_litellm_tools()
        names = [t["function"]["name"] for t in tool_defs]
        assert "fs__ls" in names

    def test_mcp_tools_require_approval(self):
        router, client = self._make_router_with_mcp()
        # MCP tools always require approval (contain "__")
        assert router.requires_approval("someserver__sometool") is True

    def test_builtin_non_destructive_no_approval(self):
        router, _ = self._make_router_with_mcp()
        assert router.requires_approval("search_rl_papers") is False

    def test_builtin_destructive_requires_approval(self):
        router, _ = self._make_router_with_mcp()
        assert router.requires_approval("run_bash") is True

    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self):
        router, client = self._make_router_with_mcp()

        mock_conn = AsyncMock()
        mock_conn.list_tools = AsyncMock(return_value=[
            {"name": "ping", "description": "Ping", "inputSchema": {"type": "object", "properties": {}}}
        ])
        mock_conn.call_tool = AsyncMock(return_value="pong from server")
        await client._register_tools("net", mock_conn)

        result = await router.execute("net__ping", {})
        assert result == "pong from server"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_returns_error(self):
        router, _ = self._make_router_with_mcp()
        result = await router.execute("totally__unknown", {})
        assert "Unknown tool" in result or "not found" in result.lower()


# ---------------------------------------------------------------------------
# to_litellm_tools schema shape
# ---------------------------------------------------------------------------

class TestMCPSchemaConversion:
    @pytest.mark.asyncio
    async def test_schema_uses_input_schema(self):
        client = MCPClient({})
        mock_conn = AsyncMock()
        mock_conn.list_tools = AsyncMock(return_value=[
            {
                "name": "do_thing",
                "description": "Does a thing",
                "inputSchema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            }
        ])
        await client._register_tools("srv", mock_conn)
        tools = client.to_litellm_tools()
        assert len(tools) == 1
        fn = tools[0]["function"]
        assert fn["name"] == "srv__do_thing"
        assert "[MCP:srv]" in fn["description"]
        assert fn["parameters"]["properties"]["x"]["type"] == "integer"

    @pytest.mark.asyncio
    async def test_schema_fallback_when_no_input_schema(self):
        client = MCPClient({})
        mock_conn = AsyncMock()
        mock_conn.list_tools = AsyncMock(return_value=[
            {"name": "bare_tool", "description": "No schema"}
        ])
        await client._register_tools("srv", mock_conn)
        tools = client.to_litellm_tools()
        fn = tools[0]["function"]
        assert fn["parameters"]["type"] == "object"
