from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.reverse_code_door import (  # noqa: E402
    EpisodeConfig,
    ReverseCodeDoorEnv,
    TemporalAction,
    evaluate_policy,
    linear_policy,
    rewind_policy,
)


def test_branch_requires_ago_greater_than_zero() -> None:
    env = ReverseCodeDoorEnv(config=EpisodeConfig(budget=8))
    env.reset(seed=0)
    env.step(TemporalAction(command="forward"))

    with pytest.raises(ValueError, match="ago > 0"):
        env.step(TemporalAction(kind="branch", instruction="retry", ago=0))


def test_rewind_policy_beats_linear_under_tight_budget() -> None:
    cfg = EpisodeConfig(budget=6)
    linear = evaluate_policy(linear_policy, episodes=100, seed=7, config=cfg)
    rewind = evaluate_policy(rewind_policy, episodes=100, seed=7, config=cfg)

    assert linear["success_rate"] == 0.0
    assert rewind["success_rate"] > 0.95
    assert rewind["branch_rate"] > 0.95


def test_branch_switches_active_timeline() -> None:
    env = ReverseCodeDoorEnv(config=EpisodeConfig(budget=8))
    obs = env.reset(seed=2)
    first_timeline = obs["active_timeline_id"]

    env.step(TemporalAction(command="forward"))
    env.step(TemporalAction(command="forward"))
    env.step(TemporalAction(command="forward"))
    env.step(TemporalAction(command="inspect"))
    branched = env.step(TemporalAction(kind="branch", instruction="Use code", ago=3))

    assert branched["active_timeline_id"] != first_timeline
    assert branched["last_branch_event"]["from_timeline_id"] == first_timeline
    assert branched["timeline_status"] == "active"


def test_shortest_path_reward_prefers_shorter_final_path() -> None:
    cfg = EpisodeConfig(
        budget=12,
        step_cost=0.0,
        success_reward=1.0,
        optimal_final_path_length=2,
        final_path_penalty=0.2,
    )
    env = ReverseCodeDoorEnv(config=cfg)
    env.reset(seed=3)

    # Long linear solve: scout oracle and walk back, then unlock.
    env.step(TemporalAction(command="forward"))
    env.step(TemporalAction(command="forward"))
    env.step(TemporalAction(command="forward"))
    obs = env.step(TemporalAction(command="inspect"))
    code = obs["visible_code"]
    env.step(TemporalAction(command="backward"))
    env.step(TemporalAction(command="backward"))
    obs = env.step(TemporalAction(command="unlock", unlock_code=code))
    linear_reward = obs["reward"]

    # Short final path solve via rewind.
    env.reset(seed=3)
    env.step(TemporalAction(command="forward"))
    env.step(TemporalAction(command="forward"))
    env.step(TemporalAction(command="forward"))
    obs = env.step(TemporalAction(command="inspect"))
    env.step(
        TemporalAction(
            kind="branch",
            instruction=f"Use code {obs['visible_code']}",
            ago=3,
        )
    )
    obs = env.step(TemporalAction(command="unlock"))
    rewind_reward = obs["reward"]

    assert obs["done"] is True
    assert rewind_reward > linear_reward
