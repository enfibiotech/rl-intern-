"""
agent_loop.py — the core agentic loop for RL Intern.

Mirrors the ml-intern submission_loop design:
  • submission_queue  → incoming Operations (user input, approvals, …)
  • event_queue       → outgoing Events consumed by the CLI / UI
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import litellm  # type: ignore

from .context_manager import ContextManager
from .doom_loop import DoomLoopDetector
from .events import Event
from .session import Session


# ---------------------------------------------------------------------------
# Operations (CLI → loop)
# ---------------------------------------------------------------------------

class OpType(str, Enum):
    USER_INPUT = "user_input"
    EXEC_APPROVAL = "exec_approval"
    INTERRUPT = "interrupt"
    COMPACT = "compact"
    UNDO = "undo"
    SHUTDOWN = "shutdown"


@dataclass
class Operation:
    type: OpType
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------

class AgentLoop:
    def __init__(self, session: Session, max_iterations: int = 300, stream: bool = True) -> None:
        self.session = session
        self.max_iterations = max_iterations
        self.stream = stream

        self.submission_queue: asyncio.Queue[Operation] = asyncio.Queue()
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()

        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._undo_stack: list[list] = []
        self._running = False

    # ------------------------------------------------------------------
    # Public: enqueue operations
    # ------------------------------------------------------------------

    async def send_user_input(self, text: str) -> None:
        await self.submission_queue.put(Operation(OpType.USER_INPUT, {"text": text}))

    async def approve(self, call_id: str, approved: bool) -> None:
        await self.submission_queue.put(
            Operation(OpType.EXEC_APPROVAL, {"call_id": call_id, "approved": approved})
        )

    async def interrupt(self) -> None:
        await self.submission_queue.put(Operation(OpType.INTERRUPT))

    async def shutdown(self) -> None:
        await self.submission_queue.put(Operation(OpType.SHUTDOWN))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit(self, event: Event) -> None:
        await self.event_queue.put(event)

    async def _wait_for_approval(self, call_id: str) -> bool:
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending_approvals[call_id] = fut
        return await fut

    # ------------------------------------------------------------------
    # Main submission loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._running = True
        await self._emit(Event.ready())

        while self._running:
            op = await self.submission_queue.get()

            if op.type == OpType.SHUTDOWN:
                await self._emit(Event.shutdown())
                self._running = False

            elif op.type == OpType.EXEC_APPROVAL:
                call_id = op.data.get("call_id", "")
                approved = op.data.get("approved", False)
                fut = self._pending_approvals.pop(call_id, None)
                if fut and not fut.done():
                    fut.set_result(approved)

            elif op.type == OpType.INTERRUPT:
                await self._emit(Event.interrupted())

            elif op.type == OpType.COMPACT:
                tb, ta = self.session.context.compact()
                await self._emit(Event.compacted(tb, ta))

            elif op.type == OpType.UNDO:
                if self._undo_stack:
                    snap = self._undo_stack.pop()
                    self.session.restore(snap)
                    await self._emit(Event(type="undo_complete", data={}))  # type: ignore[arg-type]

            elif op.type == OpType.USER_INPUT:
                snap = self.session.snapshot()
                self._undo_stack.append(snap)
                await self._emit(Event.processing())
                await self._run_agent(op.data["text"])
                await self._emit(Event.turn_complete())
                await self._emit(Event.ready())

    # ------------------------------------------------------------------
    # Agentic loop
    # ------------------------------------------------------------------

    async def _run_agent(self, user_text: str) -> None:
        ctx = self.session.context
        router = self.session.router
        doom = DoomLoopDetector()

        ctx.add_message("user", user_text)

        tools = router.to_litellm_tools()

        for iteration in range(self.max_iterations):
            # Auto-compact if needed
            if ctx.should_compact():
                tb, ta = ctx.compact()
                await self._emit(Event.compacted(tb, ta))

            # Doom-loop guard
            if doom.is_looping():
                ctx.add_message("user", doom.corrective_prompt())
                doom.reset()

            # ── LLM call ──────────────────────────────────────────────
            assistant_text = ""
            tool_calls_raw: list[dict] = []

            try:
                if self.stream:
                    response = await litellm.acompletion(
                        model=self.session.model_name,
                        messages=ctx.get_messages(),
                        tools=tools,
                        stream=True,
                    )
                    async for chunk in response:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            assistant_text += delta.content
                            await self._emit(Event.assistant_chunk(delta.content))
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                while len(tool_calls_raw) <= idx:
                                    tool_calls_raw.append(
                                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                                    )
                                if tc.id:
                                    tool_calls_raw[idx]["id"] = tc.id
                                if tc.function.name:
                                    tool_calls_raw[idx]["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    tool_calls_raw[idx]["function"]["arguments"] += tc.function.arguments
                    await self._emit(Event.assistant_stream_end())
                else:
                    response = await litellm.acompletion(
                        model=self.session.model_name,
                        messages=ctx.get_messages(),
                        tools=tools,
                        stream=False,
                    )
                    msg = response.choices[0].message
                    assistant_text = msg.content or ""
                    if msg.tool_calls:
                        tool_calls_raw = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in msg.tool_calls
                        ]

            except Exception as exc:
                await self._emit(Event.error(str(exc)))
                return

            if assistant_text:
                await self._emit(Event.assistant_message(assistant_text))

            # No tool calls → done
            if not tool_calls_raw:
                ctx.add_message("assistant", assistant_text)
                return

            # ── Tool execution ────────────────────────────────────────
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                    for tc in tool_calls_raw
                ],
            }
            ctx.add_tool_call_message(assistant_msg)

            for tc in tool_calls_raw:
                tool_name = tc["function"]["name"]
                call_id = tc["id"]

                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}

                doom.record(tool_name, args)
                await self._emit(Event.tool_call(tool_name, args))

                # Approval gate
                if router.requires_approval(tool_name):
                    await self._emit(Event.approval_required(tool_name, args, call_id))
                    approved = await self._wait_for_approval(call_id)
                    if not approved:
                        result = f"Tool '{tool_name}' was denied by the user."
                        ctx.add_tool_result(call_id, tool_name, result)
                        await self._emit(Event.tool_output(tool_name, result))
                        continue

                # Execute
                result = await router.execute(tool_name, args)
                ctx.add_tool_result(call_id, tool_name, result)
                await self._emit(Event.tool_output(tool_name, result))

        await self._emit(Event.error(f"Reached max iterations ({self.max_iterations})."))
