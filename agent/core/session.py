"""
Session — owns a ContextManager and a ToolRouter for one conversation.
"""
from __future__ import annotations

from .context_manager import ContextManager
from .tool_router import ToolRouter
from .tools import ToolSpec


SYSTEM_PROMPT = """\
You are **RL Intern** 🤖 — an expert Reinforcement Learning engineer powered by Claude.

Your job is to autonomously:
1. **Research** RL algorithms and environments by reading arXiv papers and documentation.
2. **Design** experiments: choose the right algorithm, hyperparameters, and environment.
3. **Implement** training scripts using Stable-Baselines3, CleanRL, or TRL.
4. **Train** agents by running code in the sandbox.
5. **Evaluate** trained models and interpret results.
6. **Ship** models to the Hugging Face Hub with good documentation.

## Guidelines
- Always start complex tasks with `create_plan` to outline your approach.
- Prefer `generate_training_script` over writing scripts from scratch.
- Use `inspect_environment` before training to understand obs/action spaces.
- Search papers with `search_rl_papers` before implementing a novel algorithm.
- Ask for user approval before uploading to HF Hub or running destructive bash commands.
- When training finishes, always call `evaluate_model` to report results.
- Be concise but thorough. Show your reasoning.
"""


class Session:
    def __init__(
        self,
        model_name: str,
        token_limit: int = 170_000,
        extra_tools: list[ToolSpec] | None = None,
        system_prompt_extra: str = "",
    ) -> None:
        self.model_name = model_name
        self.context = ContextManager(token_limit=token_limit)
        self.router = ToolRouter(extra_tools=extra_tools)

        full_system = SYSTEM_PROMPT
        if system_prompt_extra:
            full_system += f"\n\n{system_prompt_extra}"
        self.context.add_message("system", full_system)

    def snapshot(self):
        return self.context.snapshot()

    def restore(self, snap):
        self.context.restore(snap)
