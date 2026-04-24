"""
cli.py — RL Intern command-line interface.

Mirrors the ml-intern CLI:
  • Interactive REPL mode (default)
  • Headless / single-prompt mode
  • Streaming output with Rich formatting
  • Approval gates for sandbox/destructive tools
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text
from rich import print as rprint

from .core.agent_loop import AgentLoop, OpType
from .core.events import EventType
from .core.session import Session

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

app = typer.Typer(
    name="rl-intern",
    help="🤖 RL Intern — autonomous RL engineer powered by Claude.",
    add_completion=False,
)

console = Console()

LOGO = """
[bold cyan]
  ██████╗ ██╗         ██╗███╗   ██╗████████╗███████╗██████╗ ███╗   ██╗
  ██╔══██╗██║         ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗████╗  ██║
  ██████╔╝██║         ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝██╔██╗ ██║
  ██╔══██╗██║         ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██║╚██╗██║
  ██║  ██║███████╗    ██║██║ ╚████║   ██║   ███████╗██║  ██║██║ ╚████║
  ╚═╝  ╚═╝╚══════╝    ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝
[/bold cyan]
[dim]  🤖 Autonomous RL Engineer · Reads papers · Trains agents · Ships models[/dim]
"""


def _check_env() -> str:
    """Validate required environment variables and return the Anthropic key."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        console.print(
            Panel(
                "[bold red]ANTHROPIC_API_KEY is not set.[/bold red]\n\n"
                "Create a [cyan].env[/cyan] file with:\n"
                "  ANTHROPIC_API_KEY=sk-ant-...\n"
                "  HF_TOKEN=hf_...\n"
                "  GITHUB_TOKEN=ghp_...",
                title="Missing API Key",
                border_style="red",
            )
        )
        raise typer.Exit(1)
    return key


def _load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Event consumer (runs in main thread, reads from event_queue)
# ---------------------------------------------------------------------------

async def _consume_events(
    loop: AgentLoop,
    auto_approve: bool,
    stream_buffer: list[str],
) -> None:
    """Drain the event_queue and render events to the terminal."""
    current_stream = []

    while True:
        event = await loop.event_queue.get()

        if event.type == EventType.READY:
            break  # turn finished

        elif event.type == EventType.SHUTDOWN:
            break

        elif event.type == EventType.PROCESSING:
            console.print()

        elif event.type == EventType.ASSISTANT_CHUNK:
            text = event.data.get("text", "")
            current_stream.append(text)
            # Print inline without newline
            sys.stdout.write(text)
            sys.stdout.flush()

        elif event.type == EventType.ASSISTANT_STREAM_END:
            full = "".join(current_stream)
            stream_buffer.append(full)
            current_stream = []
            sys.stdout.write("\n")
            sys.stdout.flush()

        elif event.type == EventType.ASSISTANT_MESSAGE:
            text = event.data.get("text", "")
            if text and not stream_buffer:
                console.print(Markdown(text))

        elif event.type == EventType.TOOL_CALL:
            name = event.data.get("name", "")
            args = event.data.get("args", {})
            args_preview = json.dumps(args, ensure_ascii=False)
            if len(args_preview) > 120:
                args_preview = args_preview[:117] + "..."
            console.print(
                f"[dim]⚙  [bold]{name}[/bold]  {args_preview}[/dim]"
            )

        elif event.type == EventType.TOOL_OUTPUT:
            name = event.data.get("name", "")
            result = event.data.get("result", "")
            preview = result[:300].replace("\n", " ") if result else ""
            console.print(f"[dim]   ↳ {preview}[/dim]")

        elif event.type == EventType.TOOL_LOG:
            console.print(f"[dim italic]{event.data.get('message', '')}[/dim italic]")

        elif event.type == EventType.APPROVAL_REQUIRED:
            tool_name = event.data.get("tool_name", "")
            args = event.data.get("args", {})
            call_id = event.data.get("call_id", "")

            console.print()
            console.print(
                Panel(
                    f"[bold yellow]Tool:[/bold yellow] {tool_name}\n"
                    f"[bold yellow]Args:[/bold yellow] {json.dumps(args, indent=2)}",
                    title="⚠️  Approval Required",
                    border_style="yellow",
                )
            )

            if auto_approve:
                console.print("[dim]Auto-approving (--auto-approve flag).[/dim]")
                await loop.approve(call_id, True)
            else:
                approved = Confirm.ask("Allow this action?", default=False)
                await loop.approve(call_id, approved)

        elif event.type == EventType.TURN_COMPLETE:
            break

        elif event.type == EventType.ERROR:
            msg = event.data.get("message", "Unknown error")
            console.print(f"\n[bold red]Error:[/bold red] {msg}")
            break

        elif event.type == EventType.COMPACTED:
            tb = event.data.get("tokens_before", 0)
            ta = event.data.get("tokens_after", 0)
            console.print(
                f"[dim]📦 Context compacted: {tb:,} → {ta:,} tokens[/dim]"
            )

        elif event.type == EventType.INTERRUPTED:
            console.print("\n[yellow]⚡ Interrupted.[/yellow]")
            break


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@app.command()
def main(
    prompt: Optional[str] = typer.Argument(None, help="One-shot prompt (headless mode)"),
    model: str = typer.Option(
        "anthropic/claude-sonnet-4-20250514",
        "--model", "-m",
        help="LiteLLM model string",
    ),
    max_iterations: int = typer.Option(300, "--max-iterations", help="Max agentic loop iterations"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable token streaming"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Auto-approve all tool executions"),
    config: str = typer.Option(
        str(Path(__file__).parent.parent / "configs" / "main_agent_config.json"),
        "--config", "-c",
        help="Path to agent config JSON",
    ),
) -> None:
    """🤖 RL Intern — autonomous RL engineer."""
    _check_env()

    cfg = _load_config(config)
    effective_model = cfg.get("model_name", model)
    effective_max_iter = cfg.get("max_iterations", max_iterations)
    effective_stream = not no_stream and cfg.get("stream", True)
    token_limit = cfg.get("context_limit_tokens", 170_000)
    system_extra = cfg.get("system_prompt_extra", "")

    console.print(LOGO)

    session = Session(
        model_name=effective_model,
        token_limit=token_limit,
        system_prompt_extra=system_extra,
    )
    agent = AgentLoop(session, max_iterations=effective_max_iter, stream=effective_stream)

    if prompt:
        # ── Headless / single-shot mode ──────────────────────────────
        asyncio.run(_headless(agent, prompt, auto_approve))
    else:
        # ── Interactive REPL ─────────────────────────────────────────
        console.print(
            "[dim]Type your RL task. "
            "[bold]/help[/bold] for commands, "
            "[bold]/quit[/bold] to exit.[/dim]\n"
        )
        asyncio.run(_interactive(agent, auto_approve))


async def _headless(agent: AgentLoop, prompt: str, auto_approve: bool) -> None:
    loop_task = asyncio.create_task(agent.run())
    await agent.send_user_input(prompt)
    buf: list[str] = []
    await _consume_events(agent, auto_approve, buf)
    await agent.shutdown()
    await loop_task


async def _interactive(agent: AgentLoop, auto_approve: bool) -> None:
    loop_task = asyncio.create_task(agent.run())

    # Drain the initial READY event
    first_event = await agent.event_queue.get()
    assert first_event.type == EventType.READY

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: Prompt.ask("[bold green]>[/bold green]")
            )
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # ── Built-in slash commands ──────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye! 👋[/dim]")
            break

        if user_input.lower() == "/help":
            _print_help()
            continue

        if user_input.lower() == "/compact":
            await agent.submission_queue.put(__import__("agent.core.agent_loop", fromlist=["Operation"]).Operation(
                __import__("agent.core.agent_loop", fromlist=["OpType"]).OpType.COMPACT
            ))
            console.print("[dim]Compaction requested.[/dim]")
            continue

        if user_input.lower() == "/undo":
            from .core.agent_loop import Operation, OpType
            await agent.submission_queue.put(Operation(OpType.UNDO))
            console.print("[dim]Undo requested.[/dim]")
            continue

        # ── Normal user message ──────────────────────────────────────
        await agent.send_user_input(user_input)
        buf: list[str] = []
        await _consume_events(agent, auto_approve, buf)

    await agent.shutdown()
    try:
        await asyncio.wait_for(loop_task, timeout=2.0)
    except asyncio.TimeoutError:
        loop_task.cancel()


def _print_help() -> None:
    console.print(
        Panel(
            "[bold cyan]RL Intern Commands[/bold cyan]\n\n"
            "  [bold]/help[/bold]       Show this message\n"
            "  [bold]/quit[/bold]       Exit rl-intern\n"
            "  [bold]/compact[/bold]    Manually compact the context window\n"
            "  [bold]/undo[/bold]       Undo the last message\n\n"
            "[bold cyan]Example prompts[/bold cyan]\n\n"
            "  • train a PPO agent on CartPole-v1 for 200k steps\n"
            "  • search arXiv for recent offline RL papers\n"
            "  • find MuJoCo environments suitable for SAC\n"
            "  • generate a DQN training script for LunarLander-v2\n"
            "  • upload my model at ./models/ppo_cartpole to HuggingFace\n"
            "  • evaluate model at ./models/best_model on CartPole-v1\n",
            title="Help",
            border_style="cyan",
        )
    )
