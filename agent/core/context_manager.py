"""
ContextManager — maintains the conversation history and handles auto-compaction
when the token count approaches the configured limit.
"""
from __future__ import annotations

import json
from typing import Any


Message = dict[str, Any]


class ContextManager:
    """Stores messages and compacts them when the context gets too long."""

    def __init__(self, token_limit: int = 170_000) -> None:
        self.token_limit = token_limit
        self._messages: list[Message] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: Any) -> None:
        self._messages.append({"role": role, "content": content})

    def add_tool_call_message(self, assistant_msg: dict) -> None:
        """Add the raw assistant message (which may contain tool_calls)."""
        self._messages.append(assistant_msg)

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": content,
            }
        )

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def estimate_tokens(self) -> int:
        """Very rough token estimate: ~4 chars per token."""
        raw = json.dumps(self._messages)
        return len(raw) // 4

    def should_compact(self) -> bool:
        return self.estimate_tokens() > self.token_limit

    def compact(self) -> tuple[int, int]:
        """
        Compact the context by summarising older messages.
        Returns (tokens_before, tokens_after).
        """
        tokens_before = self.estimate_tokens()
        if len(self._messages) <= 4:
            return tokens_before, tokens_before

        # Keep the system message (index 0) + last 8 messages.
        # In a real impl you'd call the LLM to summarise the middle.
        keep_head = self._messages[:1]
        keep_tail = self._messages[-8:]

        summary_text = (
            "[Context compacted: earlier messages were summarised to save space. "
            "Continue from the current state.]"
        )
        summary_msg: Message = {"role": "user", "content": summary_text}

        self._messages = keep_head + [summary_msg] + keep_tail
        tokens_after = self.estimate_tokens()
        return tokens_before, tokens_after

    def snapshot(self) -> list[Message]:
        """Return a deep copy for undo purposes."""
        return json.loads(json.dumps(self._messages))

    def restore(self, snapshot: list[Message]) -> None:
        self._messages = snapshot
