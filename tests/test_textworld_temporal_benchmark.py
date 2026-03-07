from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.textworld_temporal import (  # noqa: E402
    TemporalTextWorldEnv,
    TextWorldGenerationConfig,
    TextWorldTemporalAction,
    create_textworld_world,
    create_textworld_worlds,
)


class FakeTextWorldBackend:
    """Small deterministic backend for testing rewind-by-replay logic."""

    def __init__(self) -> None:
        self._pos = 0

    def reset(self, seed=None):
        self._pos = 0
        return ("you are at room 0", 0.0, False, {"seed": seed})

    def step(self, command: str):
        if command == "east":
            self._pos += 1
        elif command == "west":
            self._pos -= 1
        elif command == "finish" and self._pos == 1:
            return ("you win", 1.0, True, {"pos": self._pos})
        return (f"you are at room {self._pos}", float(self._pos), False, {"pos": self._pos})


def _env() -> TemporalTextWorldEnv:
    return TemporalTextWorldEnv(backend_factory=FakeTextWorldBackend)


def test_branch_requires_ago_gt_zero() -> None:
    env = _env()
    env.reset(seed=0)
    env.step(TextWorldTemporalAction(command="east"))

    with pytest.raises(ValueError, match="ago > 0"):
        env.step(TextWorldTemporalAction(kind="branch", instruction="retry", ago=0))


def test_branch_requires_ago_within_step_count() -> None:
    env = _env()
    env.reset(seed=0)
    env.step(TextWorldTemporalAction(command="east"))

    with pytest.raises(ValueError, match="cannot exceed"):
        env.step(TextWorldTemporalAction(kind="branch", instruction="retry", ago=2))


def test_branch_replays_and_switches_active_timeline() -> None:
    env = _env()
    reset_obs = env.reset(seed=3)
    first_timeline_id = reset_obs["active_timeline_id"]

    env.step(TextWorldTemporalAction(command="east"))
    env.step(TextWorldTemporalAction(command="east"))

    branched = env.step(TextWorldTemporalAction(kind="branch", instruction="go finish", ago=1))

    assert branched["active_timeline_id"] != first_timeline_id
    assert branched["current_step"] == 1
    assert branched["feedback"] == "you are at room 1"
    assert branched["timeline_status"] == "active"


def test_abandon_ends_episode() -> None:
    env = _env()
    env.reset(seed=1)

    obs = env.step(TextWorldTemporalAction(kind="abandon"))

    assert obs["done"] is True
    assert obs["timeline_status"] == "abandoned"


def test_step_reward_uses_score_delta_plus_step_cost() -> None:
    env = TemporalTextWorldEnv(
        backend_factory=FakeTextWorldBackend,
        config=None,
    )
    env.reset(seed=0)

    obs = env.step(TextWorldTemporalAction(command="east"))

    # score changed from 0.0 -> 1.0, default step_cost is 0.0
    assert obs["reward"] == 1.0


def test_create_textworld_world_requires_tw_make(monkeypatch) -> None:
    monkeypatch.setattr("benchmarks.textworld_temporal.shutil.which", lambda _: None)

    with pytest.raises(ImportError, match="tw-make"):
        create_textworld_world(world_name="demo")


def test_create_textworld_world_uses_expected_command(monkeypatch, tmp_path) -> None:
    calls = []

    def _fake_run(cmd, check, capture_output, text):
        calls.append((cmd, check, capture_output, text))
        return None

    monkeypatch.setattr("benchmarks.textworld_temporal.shutil.which", lambda _: "/usr/local/bin/tw-make")
    monkeypatch.setattr("benchmarks.textworld_temporal.subprocess.run", _fake_run)

    path = create_textworld_world(
        world_name="demo",
        generation=TextWorldGenerationConfig(
            output_dir=str(tmp_path),
            world_size=7,
            nb_objects=4,
            quest_length=3,
            theme="castle",
            seed=11,
            force=True,
        ),
    )

    assert path.endswith("demo.z8")
    assert len(calls) == 1
    cmd = calls[0][0]
    assert cmd[0] == "/usr/local/bin/tw-make"
    assert cmd[1] == "custom"
    assert "--world-size" in cmd
    assert "--nb-objects" in cmd
    assert "--quest-length" in cmd
    assert "--theme" in cmd
    assert "--seed" in cmd
    assert "--output" in cmd
    assert "-f" in cmd


def test_create_textworld_worlds_increments_seed(monkeypatch, tmp_path) -> None:
    calls = []

    def _fake_run(cmd, check, capture_output, text):
        calls.append(cmd)
        return None

    monkeypatch.setattr("benchmarks.textworld_temporal.shutil.which", lambda _: "/usr/local/bin/tw-make")
    monkeypatch.setattr("benchmarks.textworld_temporal.subprocess.run", _fake_run)

    worlds = create_textworld_worlds(
        prefix="curriculum",
        count=3,
        generation=TextWorldGenerationConfig(output_dir=str(tmp_path), seed=5),
    )

    assert len(worlds) == 3
    assert worlds[0].endswith("curriculum_0.z8")
    assert worlds[2].endswith("curriculum_2.z8")
    assert calls[0][calls[0].index("--seed") + 1] == "5"
    assert calls[1][calls[1].index("--seed") + 1] == "6"
    assert calls[2][calls[2].index("--seed") + 1] == "7"
