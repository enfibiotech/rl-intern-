"""
examples/custom_tool.py
=======================
Example of adding a custom tool to RL Intern — a W&B run inspector.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.core.tools import ToolSpec
from agent.core.session import Session
from agent.core.agent_loop import AgentLoop
from agent.core.events import EventType


# ---------------------------------------------------------------------------
# Custom tool: fetch W&B run metrics
# ---------------------------------------------------------------------------

async def handle_fetch_wandb_runs(project: str, limit: int = 5) -> str:
    """Fetch recent W&B runs from a project."""
    try:
        import wandb  # type: ignore

        api = wandb.Api()
        runs = api.runs(project, per_page=limit)
        lines = []
        for run in runs:
            lines.append(
                f"• {run.name}  state={run.state}  "
                f"mean_reward={run.summary.get('eval/mean_reward', 'N/A')}"
            )
        return "\n".join(lines) if lines else "No runs found."
    except ImportError:
        return "wandb not installed. Run: pip install wandb"
    except Exception as exc:
        return f"Error: {exc}"


wandb_tool = ToolSpec(
    name="fetch_wandb_runs",
    description="List recent W&B training runs for a given project.",
    parameters={
        "type": "object",
        "properties": {
            "project": {"type": "string", "description": "W&B project name, e.g. 'username/ppo-cartpole'"},
            "limit": {"type": "integer", "description": "Max runs to return", "default": 5},
        },
        "required": ["project"],
    },
    handler=lambda project, limit=5: handle_fetch_wandb_runs(project, limit),
)


# ---------------------------------------------------------------------------
# Wire it up
# ---------------------------------------------------------------------------

async def main():
    session = Session(
        model_name="anthropic/claude-sonnet-4-20250514",
        extra_tools=[wandb_tool],
    )
    agent = AgentLoop(session, max_iterations=20, stream=True)
    loop_task = asyncio.create_task(agent.run())

    await agent.event_queue.get()  # READY

    await agent.send_user_input(
        "Use the fetch_wandb_runs tool to check my project 'my-org/rl-experiments' "
        "and summarise the training results."
    )

    while True:
        event = await agent.event_queue.get()
        if event.type == EventType.ASSISTANT_CHUNK:
            print(event.data["text"], end="", flush=True)
        elif event.type in (EventType.TURN_COMPLETE, EventType.READY, EventType.ERROR):
            print()
            break

    await agent.shutdown()
    await asyncio.wait_for(loop_task, timeout=2.0)


if __name__ == "__main__":
    asyncio.run(main())
