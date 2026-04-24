"""
tools.py — all RL Intern built-in tools.

Each ToolSpec wraps:
  • name / description / JSON-schema parameters  (fed to the LLM)
  • async handler                                  (called by ToolRouter)
  • requires_approval flag                         (sandboxed/destructive ops)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[str]]
    requires_approval: bool = False

    def to_litellm_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_subprocess(cmd: str, cwd: str | None = None, timeout: int = 120) -> str:
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode(errors="replace")
        rc = proc.returncode
        return f"[exit {rc}]\n{output}"
    except asyncio.TimeoutError:
        proc.kill()
        return f"[TIMEOUT after {timeout}s]"


def _workspace() -> Path:
    ws = Path(os.getenv("RL_INTERN_WORKSPACE", Path.home() / "rl-intern-workspace"))
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

# ── 1. Search arXiv for RL papers ──────────────────────────────────────────

async def handle_search_rl_papers(query: str, max_results: int = 8) -> str:
    try:
        import arxiv  # type: ignore
    except ImportError:
        return "arxiv package not installed. Run: pip install arxiv"

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    results = []
    for r in client.results(search):
        results.append(
            f"**{r.title}**\n"
            f"  Authors: {', '.join(str(a) for a in r.authors[:3])}\n"
            f"  Published: {r.published.date()}\n"
            f"  arXiv ID: {r.entry_id.split('/')[-1]}\n"
            f"  URL: {r.pdf_url}\n"
            f"  Abstract: {textwrap.shorten(r.summary, width=300)}\n"
        )
    return "\n---\n".join(results) if results else "No results found."


# ── 2. Read / summarise an arXiv paper ────────────────────────────────────

async def handle_read_rl_paper(arxiv_id: str) -> str:
    try:
        import arxiv  # type: ignore
    except ImportError:
        return "arxiv package not installed."

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search), None)
    if paper is None:
        return f"Paper {arxiv_id} not found."

    return (
        f"# {paper.title}\n\n"
        f"**Authors:** {', '.join(str(a) for a in paper.authors)}\n"
        f"**Published:** {paper.published.date()}\n"
        f"**Categories:** {', '.join(paper.categories)}\n"
        f"**PDF:** {paper.pdf_url}\n\n"
        f"## Abstract\n{paper.summary}\n\n"
        f"## Comments\n{paper.comment or 'N/A'}\n"
    )


# ── 3. List / inspect Gymnasium environments ──────────────────────────────

async def handle_list_rl_environments(filter_str: str = "") -> str:
    try:
        import gymnasium as gym  # type: ignore
    except ImportError:
        return "gymnasium not installed. Run: pip install gymnasium"

    envs = list(gym.envs.registry.keys())
    if filter_str:
        envs = [e for e in envs if filter_str.lower() in e.lower()]
    envs.sort()
    if not envs:
        return f"No environments matching '{filter_str}'."
    return f"Found {len(envs)} environments:\n" + "\n".join(f"  • {e}" for e in envs[:60])


async def handle_inspect_environment(env_id: str) -> str:
    try:
        import gymnasium as gym  # type: ignore
    except ImportError:
        return "gymnasium not installed."

    try:
        env = gym.make(env_id)
        info = (
            f"**Environment:** {env_id}\n"
            f"**Observation space:** {env.observation_space}\n"
            f"**Action space:** {env.action_space}\n"
            f"**Reward range:** {env.reward_range}\n"
            f"**Spec max_episode_steps:** {getattr(env.spec, 'max_episode_steps', 'N/A')}\n"
        )
        env.close()
        return info
    except Exception as exc:
        return f"Error inspecting {env_id}: {exc}"


# ── 4. Search RL documentation ────────────────────────────────────────────

_RL_DOCS: dict[str, str] = {
    "gymnasium": "https://gymnasium.farama.org",
    "stable-baselines3": "https://stable-baselines3.readthedocs.io",
    "cleanrl": "https://docs.cleanrl.dev",
    "trl": "https://huggingface.co/docs/trl",
    "rllib": "https://docs.ray.io/en/latest/rllib",
    "sb3": "https://stable-baselines3.readthedocs.io",
    "pettingzoo": "https://pettingzoo.farama.org",
}

async def handle_search_rl_docs(library: str, query: str) -> str:
    lib_key = library.lower().replace(" ", "-")
    base_url = _RL_DOCS.get(lib_key, _RL_DOCS.get(lib_key.split("-")[0], None))
    url_hint = f"\nDocs homepage: {base_url}" if base_url else ""

    # Web search via subprocess curl + DuckDuckGo
    search_query = f"{library} {query} site:{base_url}" if base_url else f"{library} RL {query}"
    cmd = (
        f"curl -sG 'https://api.duckduckgo.com/' "
        f"--data-urlencode 'q={search_query}' "
        f"--data 'format=json&no_html=1&skip_disambig=1' "
        f"| python3 -c \""
        f"import sys,json; d=json.load(sys.stdin); "
        f"[print(r.get('FirstURL',''), r.get('Text','')) for r in d.get('RelatedTopics',[])[:5]]\""
    )
    result = await _run_subprocess(cmd, timeout=15)
    return f"Documentation search for '{query}' in {library}:{url_hint}\n\n{result}"


# ── 5. Generate RL training script ────────────────────────────────────────

_ALGORITHM_TEMPLATES: dict[str, str] = {
    "ppo": """
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# ── Config ──────────────────────────────────────────────────────────────
ENV_ID         = "{env_id}"
TOTAL_STEPS    = {total_steps}
N_ENVS         = 4
LEARNING_RATE  = 3e-4
N_STEPS        = 2048
BATCH_SIZE     = 64
N_EPOCHS       = 10
SEED           = 42
LOG_DIR        = "./logs/{env_id_safe}_ppo"
MODEL_SAVE_DIR = "./models/{env_id_safe}_ppo"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ── Environment ─────────────────────────────────────────────────────────
vec_env  = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
eval_env = make_vec_env(ENV_ID, n_envs=1, seed=SEED + 100)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=max(10_000 // N_ENVS, 1),
    n_eval_episodes=10,
    deterministic=True,
)

# ── Model ────────────────────────────────────────────────────────────────
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
)

# ── Training ─────────────────────────────────────────────────────────────
model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback, progress_bar=True)
model.save(f"{{MODEL_SAVE_DIR}}/final_model")
print(f"\\nTraining complete! Model saved to {{MODEL_SAVE_DIR}}/final_model")
""",

    "dqn": """
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import os

ENV_ID         = "{env_id}"
TOTAL_STEPS    = {total_steps}
LEARNING_RATE  = 1e-4
BUFFER_SIZE    = 100_000
BATCH_SIZE     = 32
LEARNING_STARTS= 1_000
SEED           = 42
LOG_DIR        = "./logs/{env_id_safe}_dqn"
MODEL_SAVE_DIR = "./models/{env_id_safe}_dqn"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

env      = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    learning_starts=LEARNING_STARTS,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback, progress_bar=True)
model.save(f"{{MODEL_SAVE_DIR}}/final_model")
print(f"\\nTraining complete! Saved to {{MODEL_SAVE_DIR}}/final_model")
""",

    "sac": """
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
import os

ENV_ID         = "{env_id}"
TOTAL_STEPS    = {total_steps}
LEARNING_RATE  = 3e-4
BUFFER_SIZE    = 1_000_000
BATCH_SIZE     = 256
SEED           = 42
LOG_DIR        = "./logs/{env_id_safe}_sac"
MODEL_SAVE_DIR = "./models/{env_id_safe}_sac"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

env      = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
)

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    verbose=1,
    tensorboard_log=LOG_DIR,
    seed=SEED,
)

model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback, progress_bar=True)
model.save(f"{{MODEL_SAVE_DIR}}/final_model")
print(f"\\nTraining complete! Saved to {{MODEL_SAVE_DIR}}/final_model")
""",

    "td3": """
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os

ENV_ID         = "{env_id}"
TOTAL_STEPS    = {total_steps}
LEARNING_RATE  = 1e-3
BUFFER_SIZE    = 1_000_000
BATCH_SIZE     = 100
SEED           = 42
LOG_DIR        = "./logs/{env_id_safe}_td3"
MODEL_SAVE_DIR = "./models/{env_id_safe}_td3"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

env      = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID)

n_actions      = env.action_space.shape[-1]
action_noise   = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

eval_callback = EvalCallback(eval_env, best_model_save_path=MODEL_SAVE_DIR,
    log_path=LOG_DIR, eval_freq=10_000, n_eval_episodes=5, deterministic=True)

model = TD3("MlpPolicy", env, action_noise=action_noise, learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, verbose=1,
    tensorboard_log=LOG_DIR, seed=SEED)

model.learn(total_timesteps=TOTAL_STEPS, callback=eval_callback, progress_bar=True)
model.save(f"{{MODEL_SAVE_DIR}}/final_model")
print(f"\\nDone! Model saved to {{MODEL_SAVE_DIR}}/final_model")
""",
}

async def handle_generate_training_script(
    algorithm: str,
    env_id: str,
    total_steps: int = 500_000,
    save_path: str | None = None,
) -> str:
    algo = algorithm.lower()
    template = _ALGORITHM_TEMPLATES.get(algo)
    if template is None:
        available = ", ".join(_ALGORITHM_TEMPLATES.keys())
        return f"Unknown algorithm '{algorithm}'. Available: {available}"

    env_id_safe = re.sub(r"[^a-zA-Z0-9_]", "_", env_id).lower()
    script = template.format(
        env_id=env_id,
        env_id_safe=env_id_safe,
        total_steps=total_steps,
    )

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(script)
        return f"Script written to {save_path}\n\n```python\n{script}\n```"

    return f"```python\n{script}\n```"


# ── 6. Run bash command in sandbox ────────────────────────────────────────

async def handle_run_bash(command: str, working_dir: str | None = None) -> str:
    cwd = working_dir or str(_workspace())
    return await _run_subprocess(command, cwd=cwd, timeout=300)


# ── 7. Run Python code in sandbox ─────────────────────────────────────────

async def handle_run_python(code: str, working_dir: str | None = None) -> str:
    cwd = working_dir or str(_workspace())
    tmp = Path(cwd) / "_rl_intern_tmp.py"
    tmp.write_text(code)
    result = await _run_subprocess(f"python3 {tmp}", cwd=cwd, timeout=300)
    tmp.unlink(missing_ok=True)
    return result


# ── 8. Read file ───────────────────────────────────────────────────────────

async def handle_read_file(path: str, max_lines: int = 300) -> str:
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    lines = p.read_text(errors="replace").splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + f"\n\n[... truncated, {len(lines)} total lines]"
    return "\n".join(lines)


# ── 9. Write file ──────────────────────────────────────────────────────────

async def handle_write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


# ── 10. Search HuggingFace Hub for RL models ─────────────────────────────

async def handle_search_hf_rl_models(query: str, limit: int = 8) -> str:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        return "huggingface-hub not installed."

    api = HfApi()
    models = list(
        api.list_models(
            search=query,
            filter="reinforcement-learning",
            sort="downloads",
            direction=-1,
            limit=limit,
        )
    )
    if not models:
        return f"No RL models found for '{query}'."

    lines = []
    for m in models:
        lines.append(
            f"• {m.id}  (downloads: {getattr(m, 'downloads', '?')}, "
            f"likes: {getattr(m, 'likes', '?')})"
        )
    return "\n".join(lines)


# ── 11. Upload model to HuggingFace Hub ──────────────────────────────────

async def handle_upload_model_to_hf(
    local_path: str,
    repo_id: str,
    commit_message: str = "Upload RL model via rl-intern",
    repo_type: str = "model",
) -> str:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        return "huggingface-hub not installed."

    token = os.getenv("HF_TOKEN")
    if not token:
        return "HF_TOKEN environment variable not set."

    api = HfApi(token=token)
    try:
        url = api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )
        return f"✅ Uploaded to {url}"
    except Exception as exc:
        return f"Upload failed: {exc}"


# ── 12. Install Python package ────────────────────────────────────────────

async def handle_install_package(package: str) -> str:
    return await _run_subprocess(f"pip install {package} -q", timeout=120)


# ── 13. Planning tool ─────────────────────────────────────────────────────

async def handle_create_plan(steps: list[str], title: str = "RL Intern Plan") -> str:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    return f"## {title}\n\n{numbered}"


# ── 14. Evaluate a trained model ─────────────────────────────────────────

async def handle_evaluate_model(
    model_path: str,
    env_id: str,
    n_episodes: int = 10,
    algorithm: str = "ppo",
) -> str:
    algo_upper = algorithm.upper()
    code = textwrap.dedent(f"""
        import gymnasium as gym
        from stable_baselines3 import {algo_upper}
        import numpy as np

        model = {algo_upper}.load("{model_path}")
        env   = gym.make("{env_id}")
        rewards = []

        for ep in range({n_episodes}):
            obs, _ = env.reset()
            done   = False
            ep_r   = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_r += reward
                done = terminated or truncated
            rewards.append(ep_r)

        print(f"Episodes: {n_episodes}")
        print(f"Mean reward : {{np.mean(rewards):.2f}} +/- {{np.std(rewards):.2f}}")
        print(f"Min / Max   : {{np.min(rewards):.2f}} / {{np.max(rewards):.2f}}")
        env.close()
    """)
    return await handle_run_python(code)


# ── 15. Show workspace contents ───────────────────────────────────────────

async def handle_list_workspace() -> str:
    ws = _workspace()
    items = sorted(ws.rglob("*"))
    if not items:
        return f"Workspace is empty: {ws}"
    lines = [f"Workspace: {ws}"]
    for item in items[:80]:
        rel = item.relative_to(ws)
        prefix = "📁" if item.is_dir() else "📄"
        lines.append(f"  {prefix} {rel}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="search_rl_papers",
            description=(
                "Search arXiv for reinforcement learning papers. "
                "Use for finding algorithms, benchmarks, or theoretical results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (e.g. 'PPO continuous control')"},
                    "max_results": {"type": "integer", "description": "Max papers to return (default 8)", "default": 8},
                },
                "required": ["query"],
            },
            handler=lambda query, max_results=8: handle_search_rl_papers(query, max_results),
        ),
        ToolSpec(
            name="read_rl_paper",
            description="Fetch full abstract and metadata for a specific arXiv paper by its ID.",
            parameters={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string", "description": "arXiv paper ID, e.g. '1707.06347'"},
                },
                "required": ["arxiv_id"],
            },
            handler=lambda arxiv_id: handle_read_rl_paper(arxiv_id),
        ),
        ToolSpec(
            name="list_rl_environments",
            description="List available Gymnasium RL environments, optionally filtered by a substring.",
            parameters={
                "type": "object",
                "properties": {
                    "filter_str": {"type": "string", "description": "Optional filter substring (e.g. 'CartPole', 'Atari', 'MuJoCo')", "default": ""},
                },
                "required": [],
            },
            handler=lambda filter_str="": handle_list_rl_environments(filter_str),
        ),
        ToolSpec(
            name="inspect_environment",
            description="Inspect the observation space, action space, and spec of a Gymnasium environment.",
            parameters={
                "type": "object",
                "properties": {
                    "env_id": {"type": "string", "description": "Gymnasium environment ID, e.g. 'CartPole-v1'"},
                },
                "required": ["env_id"],
            },
            handler=lambda env_id: handle_inspect_environment(env_id),
        ),
        ToolSpec(
            name="search_rl_docs",
            description=(
                "Search documentation for RL libraries: gymnasium, stable-baselines3, "
                "cleanrl, trl, rllib, pettingzoo."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "library": {"type": "string", "description": "Library name, e.g. 'stable-baselines3'"},
                    "query": {"type": "string", "description": "What to look up"},
                },
                "required": ["library", "query"],
            },
            handler=lambda library, query: handle_search_rl_docs(library, query),
        ),
        ToolSpec(
            name="generate_training_script",
            description=(
                "Generate a complete RL training script using Stable-Baselines3. "
                "Algorithms: ppo, dqn, sac, td3."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "algorithm": {"type": "string", "description": "RL algorithm: ppo | dqn | sac | td3"},
                    "env_id": {"type": "string", "description": "Gymnasium environment ID"},
                    "total_steps": {"type": "integer", "description": "Training timesteps", "default": 500000},
                    "save_path": {"type": "string", "description": "Optional file path to save the script"},
                },
                "required": ["algorithm", "env_id"],
            },
            handler=lambda algorithm, env_id, total_steps=500000, save_path=None: handle_generate_training_script(
                algorithm, env_id, total_steps, save_path
            ),
        ),
        ToolSpec(
            name="run_bash",
            description="Execute a bash command in the RL Intern workspace (sandbox). Use for installs, running scripts, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "working_dir": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["command"],
            },
            handler=lambda command, working_dir=None: handle_run_bash(command, working_dir),
            requires_approval=True,
        ),
        ToolSpec(
            name="run_python",
            description="Execute Python code in the RL Intern workspace. Good for quick experiments or data analysis.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "working_dir": {"type": "string", "description": "Working directory (optional)"},
                },
                "required": ["code"],
            },
            handler=lambda code, working_dir=None: handle_run_python(code, working_dir),
            requires_approval=True,
        ),
        ToolSpec(
            name="read_file",
            description="Read the contents of a file.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                    "max_lines": {"type": "integer", "description": "Max lines to return (default 300)", "default": 300},
                },
                "required": ["path"],
            },
            handler=lambda path, max_lines=300: handle_read_file(path, max_lines),
        ),
        ToolSpec(
            name="write_file",
            description="Write content to a file (creates parent directories as needed).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
            handler=lambda path, content: handle_write_file(path, content),
            requires_approval=True,
        ),
        ToolSpec(
            name="search_hf_rl_models",
            description="Search the Hugging Face Hub for trained RL models.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, e.g. 'PPO CartPole'"},
                    "limit": {"type": "integer", "description": "Max results (default 8)", "default": 8},
                },
                "required": ["query"],
            },
            handler=lambda query, limit=8: handle_search_hf_rl_models(query, limit),
        ),
        ToolSpec(
            name="upload_model_to_hf",
            description="Upload a trained RL model folder to the Hugging Face Hub.",
            parameters={
                "type": "object",
                "properties": {
                    "local_path": {"type": "string", "description": "Local directory containing the model"},
                    "repo_id": {"type": "string", "description": "HF repo ID, e.g. 'username/ppo-cartpole'"},
                    "commit_message": {"type": "string", "description": "Commit message"},
                    "repo_type": {"type": "string", "description": "model | dataset | space", "default": "model"},
                },
                "required": ["local_path", "repo_id"],
            },
            handler=lambda local_path, repo_id, commit_message="Upload RL model", repo_type="model": handle_upload_model_to_hf(
                local_path, repo_id, commit_message, repo_type
            ),
            requires_approval=True,
        ),
        ToolSpec(
            name="install_package",
            description="Install a Python package with pip (e.g. stable-baselines3, gymnasium[classic-control]).",
            parameters={
                "type": "object",
                "properties": {
                    "package": {"type": "string", "description": "Package name/spec to install"},
                },
                "required": ["package"],
            },
            handler=lambda package: handle_install_package(package),
            requires_approval=True,
        ),
        ToolSpec(
            name="evaluate_model",
            description="Run a trained SB3 model for N episodes and report mean/std reward.",
            parameters={
                "type": "object",
                "properties": {
                    "model_path": {"type": "string", "description": "Path to saved model (without .zip)"},
                    "env_id": {"type": "string", "description": "Gymnasium environment ID"},
                    "n_episodes": {"type": "integer", "description": "Number of evaluation episodes (default 10)", "default": 10},
                    "algorithm": {"type": "string", "description": "Algorithm used: ppo | dqn | sac | td3", "default": "ppo"},
                },
                "required": ["model_path", "env_id"],
            },
            handler=lambda model_path, env_id, n_episodes=10, algorithm="ppo": handle_evaluate_model(
                model_path, env_id, n_episodes, algorithm
            ),
        ),
        ToolSpec(
            name="list_workspace",
            description="Show the contents of the RL Intern workspace directory.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=lambda: handle_list_workspace(),
        ),
        ToolSpec(
            name="create_plan",
            description="Create a numbered plan for a multi-step RL task.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Plan title"},
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered list of steps",
                    },
                },
                "required": ["steps"],
            },
            handler=lambda steps, title="RL Intern Plan": handle_create_plan(steps, title),
        ),
    ]
