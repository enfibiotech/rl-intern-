"""
Event types emitted by the RL Intern agentic loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    # lifecycle
    PROCESSING = "processing"
    READY = "ready"
    SHUTDOWN = "shutdown"
    INTERRUPTED = "interrupted"
    ERROR = "error"

    # LLM streaming
    ASSISTANT_CHUNK = "assistant_chunk"
    ASSISTANT_MESSAGE = "assistant_message"
    ASSISTANT_STREAM_END = "assistant_stream_end"

    # tools
    TOOL_CALL = "tool_call"
    TOOL_OUTPUT = "tool_output"
    TOOL_LOG = "tool_log"
    TOOL_STATE_CHANGE = "tool_state_change"

    # approvals & flow
    APPROVAL_REQUIRED = "approval_required"
    TURN_COMPLETE = "turn_complete"
    COMPACTED = "compacted"
    UNDO_COMPLETE = "undo_complete"


@dataclass
class Event:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    # ----- convenience constructors -----

    @classmethod
    def processing(cls) -> "Event":
        return cls(type=EventType.PROCESSING)

    @classmethod
    def ready(cls) -> "Event":
        return cls(type=EventType.READY)

    @classmethod
    def assistant_chunk(cls, text: str) -> "Event":
        return cls(type=EventType.ASSISTANT_CHUNK, data={"text": text})

    @classmethod
    def assistant_message(cls, text: str) -> "Event":
        return cls(type=EventType.ASSISTANT_MESSAGE, data={"text": text})

    @classmethod
    def assistant_stream_end(cls) -> "Event":
        return cls(type=EventType.ASSISTANT_STREAM_END)

    @classmethod
    def tool_call(cls, name: str, args: dict) -> "Event":
        return cls(type=EventType.TOOL_CALL, data={"name": name, "args": args})

    @classmethod
    def tool_output(cls, name: str, result: str) -> "Event":
        return cls(type=EventType.TOOL_OUTPUT, data={"name": name, "result": result})

    @classmethod
    def tool_log(cls, message: str) -> "Event":
        return cls(type=EventType.TOOL_LOG, data={"message": message})

    @classmethod
    def approval_required(cls, tool_name: str, args: dict, call_id: str) -> "Event":
        return cls(
            type=EventType.APPROVAL_REQUIRED,
            data={"tool_name": tool_name, "args": args, "call_id": call_id},
        )

    @classmethod
    def turn_complete(cls) -> "Event":
        return cls(type=EventType.TURN_COMPLETE)

    @classmethod
    def error(cls, message: str) -> "Event":
        return cls(type=EventType.ERROR, data={"message": message})

    @classmethod
    def compacted(cls, tokens_before: int, tokens_after: int) -> "Event":
        return cls(
            type=EventType.COMPACTED,
            data={"tokens_before": tokens_before, "tokens_after": tokens_after},
        )

    @classmethod
    def interrupted(cls) -> "Event":
        return cls(type=EventType.INTERRUPTED)

    @classmethod
    def shutdown(cls) -> "Event":
        return cls(type=EventType.SHUTDOWN)
