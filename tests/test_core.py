"""
Tests for RL Intern core components.
Run with: pytest tests/ -v
"""
from __future__ import annotations

import asyncio
import pytest

from agent.core.doom_loop import DoomLoopDetector
from agent.core.context_manager import ContextManager
from agent.core.tools import (
    handle_generate_training_script,
    handle_list_rl_environments,
    handle_create_plan,
    handle_write_file,
    handle_read_file,
)


# ---------------------------------------------------------------------------
# DoomLoopDetector
# ---------------------------------------------------------------------------

class TestDoomLoopDetector:
    def test_no_loop_initially(self):
        d = DoomLoopDetector()
        assert not d.is_looping()

    def test_detects_repeated_pattern(self):
        d = DoomLoopDetector()
        for _ in range(DoomLoopDetector.THRESHOLD * DoomLoopDetector.WINDOW + 2):
            d.record("run_bash", {"command": "echo hi"})
        assert d.is_looping()

    def test_reset_clears_state(self):
        d = DoomLoopDetector()
        for _ in range(30):
            d.record("run_bash", {"command": "echo hi"})
        d.reset()
        assert not d.is_looping()

    def test_varied_calls_no_loop(self):
        d = DoomLoopDetector()
        for i in range(20):
            d.record("run_bash", {"command": f"echo {i}"})
        assert not d.is_looping()


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_add_and_get_messages(self):
        ctx = ContextManager()
        ctx.add_message("user", "hello")
        msgs = ctx.get_messages()
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "hello"

    def test_should_compact_when_over_limit(self):
        ctx = ContextManager(token_limit=10)
        ctx.add_message("user", "x" * 200)
        assert ctx.should_compact()

    def test_compact_reduces_tokens(self):
        ctx = ContextManager()
        for i in range(30):
            ctx.add_message("user", f"message {i} " + "x" * 500)
            ctx.add_message("assistant", f"reply {i} " + "y" * 500)
        before = ctx.estimate_tokens()
        ctx.compact()
        after = ctx.estimate_tokens()
        assert after < before

    def test_snapshot_and_restore(self):
        ctx = ContextManager()
        ctx.add_message("user", "original")
        snap = ctx.snapshot()
        ctx.add_message("user", "extra")
        assert len(ctx.get_messages()) == 2
        ctx.restore(snap)
        assert len(ctx.get_messages()) == 1


# ---------------------------------------------------------------------------
# Tool handlers (async)
# ---------------------------------------------------------------------------

class TestGenerateTrainingScript:
    def test_ppo_script_contains_env_id(self):
        result = asyncio.run(handle_generate_training_script("ppo", "CartPole-v1"))
        assert "CartPole-v1" in result
        assert "PPO" in result

    def test_dqn_script_generation(self):
        result = asyncio.run(handle_generate_training_script("dqn", "LunarLander-v2", total_steps=100_000))
        assert "DQN" in result
        assert "LunarLander-v2" in result
        assert "100000" in result

    def test_unknown_algorithm_error(self):
        result = asyncio.run(handle_generate_training_script("unknown_algo", "CartPole-v1"))
        assert "Unknown algorithm" in result

    def test_save_to_path(self, tmp_path):
        out = tmp_path / "train.py"
        result = asyncio.run(handle_generate_training_script("ppo", "CartPole-v1", save_path=str(out)))
        assert out.exists()
        assert "CartPole-v1" in out.read_text()


class TestCreatePlan:
    def test_numbered_steps(self):
        result = asyncio.run(handle_create_plan(["Step A", "Step B", "Step C"], title="My Plan"))
        assert "1. Step A" in result
        assert "2. Step B" in result
        assert "My Plan" in result


class TestFileOps:
    def test_write_and_read(self, tmp_path):
        path = str(tmp_path / "test.txt")
        asyncio.run(handle_write_file(path, "hello world"))
        content = asyncio.run(handle_read_file(path))
        assert "hello world" in content

    def test_read_nonexistent(self):
        result = asyncio.run(handle_read_file("/nonexistent/file.txt"))
        assert "not found" in result.lower()

    def test_read_truncates_long_file(self, tmp_path):
        path = str(tmp_path / "long.txt")
        lines = "\n".join(f"line {i}" for i in range(500))
        asyncio.run(handle_write_file(path, lines))
        result = asyncio.run(handle_read_file(path, max_lines=10))
        assert "truncated" in result


class TestListEnvironments:
    def test_returns_environments(self):
        try:
            result = asyncio.run(handle_list_rl_environments())
            assert "CartPole" in result or "environments" in result
        except ImportError:
            pytest.skip("gymnasium not installed")

    def test_filter_works(self):
        try:
            result = asyncio.run(handle_list_rl_environments("CartPole"))
            assert "CartPole" in result
        except ImportError:
            pytest.skip("gymnasium not installed")
