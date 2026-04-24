"""
ToolRouter — dispatches tool-call requests to the correct handler.
Supports built-in tools and MCP server tools.
"""
from __future__ import annotations

import json
from typing import Any

from .tools import ToolSpec, create_builtin_tools


class ToolRouter:
    def __init__(self, extra_tools: list[ToolSpec] | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for spec in create_builtin_tools():
            self._tools[spec.name] = spec
        for spec in (extra_tools or []):
            self._tools[spec.name] = spec

    # ------------------------------------------------------------------

    def get_all_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def to_litellm_tools(self) -> list[dict]:
        return [spec.to_litellm_tool() for spec in self._tools.values()]

    def requires_approval(self, tool_name: str) -> bool:
        spec = self._tools.get(tool_name)
        return spec.requires_approval if spec else False

    async def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        spec = self._tools.get(tool_name)
        if spec is None:
            return f"Unknown tool: {tool_name}"
        try:
            result = await spec.handler(**args)
            return str(result)
        except TypeError as exc:
            return f"Tool argument error for '{tool_name}': {exc}"
        except Exception as exc:
            return f"Tool execution error for '{tool_name}': {exc}"
