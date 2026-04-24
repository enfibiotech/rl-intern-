"""
DoomLoopDetector — detects when the agent is stuck in a repetitive tool-call pattern
and injects a corrective prompt to break the cycle.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class DoomLoopDetector:
    """
    Tracks the last N tool calls and fires when an identical pattern repeats
    more than ``threshold`` times in a row.
    """

    WINDOW = 6
    THRESHOLD = 3

    def __init__(self) -> None:
        self._history: deque[str] = deque(maxlen=self.WINDOW * self.THRESHOLD)

    def record(self, tool_name: str, args: dict[str, Any]) -> None:
        key = f"{tool_name}:{sorted(args.items())}"
        self._history.append(key)

    def is_looping(self) -> bool:
        if len(self._history) < self.WINDOW * 2:
            return False
        items = list(self._history)
        # Check if the last WINDOW calls are identical
        last = items[-self.WINDOW :]
        for i in range(1, self.THRESHOLD):
            prev = items[-(self.WINDOW * (i + 1)) : -(self.WINDOW * i)]
            if len(prev) == self.WINDOW and last == prev:
                return True
        return False

    def corrective_prompt(self) -> str:
        return (
            "⚠️  You appear to be repeating the same tool calls without making progress. "
            "Please reconsider your approach: try a different tool, a different strategy, "
            "or let the user know if you are stuck. Do NOT repeat the same failed action."
        )

    def reset(self) -> None:
        self._history.clear()
