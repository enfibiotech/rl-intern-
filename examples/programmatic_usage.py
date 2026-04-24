"""
examples/programmatic_usage.py
================================
Use RL Intern as a Python library (no CLI needed).
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.core.session import Session
from agent.core.agent_loop import AgentLoop
from agent.core.events import EventType


async def run_rl_task(task: str) -> str:
    """Run a single RL task and return the final assistant response."""

    session = Session(model_name="anthropic/claude-sonnet-4-20250514")
    agent = AgentLoop(session, max_iterations=50, stream=False)

    # Start the loop
    loop_task = asyncio.create_task(agent.run())

    # Wait for READY
    await agent.event_queue.get()

    # Send task
    await agent.send_user_input(task)

    # Collect output
    response_parts = []
    while True:
        event = await agent.event_queue.get()
        if event.type == EventType.ASSISTANT_MESSAGE:
            response_parts.append(event.data.get("text", ""))
        elif event.type in (EventType.TURN_COMPLETE, EventType.READY, EventType.ERROR):
            break

    await agent.shutdown()
    try:
        await asyncio.wait_for(loop_task, timeout=2.0)
    except asyncio.TimeoutError:
        loop_task.cancel()

    return "\n".join(response_parts)


if __name__ == "__main__":
    result = asyncio.run(
        run_rl_task(
            "List 5 classic control Gymnasium environments suitable for DQN, "
            "with a one-sentence description of each."
        )
    )
    print(result)
