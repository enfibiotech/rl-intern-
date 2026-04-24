# 🤖 RL Intern

An autonomous Reinforcement Learning engineer that reads papers, sets up environments, trains agents, and ships RL models — powered by Claude and the Hugging Face ecosystem.

---

## Quick Start

### Installation

```bash
git clone https://github.com/your-org/rl-intern.git
cd rl-intern
uv sync
uv tool install -e .
```

#### That's it. Now `rl-intern` works from any directory:

```bash
rl-intern
```

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=<your-anthropic-api-key>
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token>
WANDB_API_KEY=<optional-wandb-api-key>
```

---

### Usage

**Interactive mode** (start a chat session):
```bash
rl-intern
```

**Headless mode** (single prompt, auto-approve):
```bash
rl-intern "train a PPO agent on CartPole-v1 and upload to HuggingFace"
```

**Options:**
```bash
rl-intern --model anthropic/claude-opus-4-6 "your prompt"
rl-intern --max-iterations 150 "your prompt"
rl-intern --no-stream "your prompt"
rl-intern --auto-approve "train DQN on LunarLander"
```

---

## What It Can Do

| Capability | Details |
|---|---|
| 📄 **Read RL Papers** | Fetches and summarises papers from arXiv (cs.LG, cs.AI, stat.ML) |
| 🌍 **Browse RL Environments** | Lists and inspects Gymnasium / PettingZoo / Atari / MuJoCo envs |
| 🛠️ **Generate Training Scripts** | PPO, SAC, DQN, DDPG, A2C, TD3 via CleanRL / SB3 / TRL |
| 🏋️ **Run Training** | Executes training in a sandboxed subprocess with live logs |
| 📊 **Log Experiments** | W&B and TensorBoard integration out of the box |
| 🤗 **HF Hub** | Search, download, and upload trained RL models & videos |
| 🔍 **Search Docs** | Queries Gymnasium, SB3, CleanRL, TRL, and RLHF documentation |
| 💻 **Write & Run Code** | Full bash + Python sandbox with approval gate for destructive ops |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         User / CLI                           │
└──────────┬───────────────────────────────────────┬───────────┘
           │ Operations                            │ Events
           ↓ (user_input, exec_approval, ...)      ↑
    submission_queue                          event_queue
           │                                       │
           ↓                                       ↓
┌──────────────────────────────────────────────────────────┐
│           submission_loop  (agent_loop.py)               │
│  ┌──────────────────────────────────────────────────┐    │
│  │  1. Receive Operation from queue                 │    │
│  │  2. Route → run_agent / compact / undo / …       │    │
│  └──────────────────────────────────────────────────┘    │
│                       ↓                                  │
│  ┌──────────────────────────────────────────────────┐    │
│  │          Handlers.run_agent()                    │    │
│  │                                                  │    │
│  │  ┌────────────────────────────────────────────┐  │    │
│  │  │   Agentic Loop  (max 300 iterations)       │  │    │
│  │  │                                            │  │    │
│  │  │   Session                                  │  │    │
│  │  │    ├─ ContextManager (message history,     │  │    │
│  │  │    │                  auto-compact 170k)   │  │    │
│  │  │    └─ ToolRouter                           │  │    │
│  │  │         ├─ RL papers  (arXiv)              │  │    │
│  │  │         ├─ RL docs    (Gym / SB3 / CleanRL)│  │    │
│  │  │         ├─ Environments (Gymnasium)        │  │    │
│  │  │         ├─ HF Hub  (models / datasets)     │  │    │
│  │  │         ├─ Training scripts generator      │  │    │
│  │  │         ├─ Sandbox  (bash / python)        │  │    │
│  │  │         ├─ File I/O                        │  │    │
│  │  │         └─ Planning                        │  │    │
│  │  │                                            │  │    │
│  │  │   DoomLoopDetector                         │  │    │
│  │  └────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Adding Custom Tools

Edit `agent/core/tools.py`:

```python
ToolSpec(
    name="my_rl_tool",
    description="What your tool does",
    parameters={
        "type": "object",
        "properties": {
            "env_id": {"type": "string", "description": "Gymnasium env ID"}
        },
        "required": ["env_id"]
    },
    handler=my_async_handler,
    requires_approval=False,
)
```

## Adding MCP Servers

Edit `configs/main_agent_config.json`:

```json
{
  "mcpServers": {
    "your-server": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": { "Authorization": "Bearer ${YOUR_TOKEN}" }
    }
  }
}
```

---

## License

Apache 2.0
