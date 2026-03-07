from __future__ import annotations

from unittest.mock import patch

import pytest

from timetravel.models import TimetravelAction
from timetravel.server.timetravel_environment import TimetravelEnvironment


def test_branch_rewinds_with_valid_ago() -> None:
    env = TimetravelEnvironment(max_rounds=5)
    env.reset()

    with patch("timetravel.server.timetravel_environment.random.choice", side_effect=["scissors", "paper"]):
        obs1 = env.step(TimetravelAction(kind="step", message="rock"))
        obs2 = env.step(TimetravelAction(kind="step", message="rock"))

    assert obs1.world_state["round_index"] == 1
    assert obs2.world_state["round_index"] == 2

    branched = env.step(TimetravelAction(kind="branch", instruction="retry from one step back", ago=1))

    assert branched.current_step == 1
    assert branched.timeline_status == "active"
    assert branched.world_state["round_index"] == 1
    assert branched.last_branch_event is not None


def test_branch_with_ago_zero_is_rejected() -> None:
    env = TimetravelEnvironment(max_rounds=5)
    env.reset()
    env.step(TimetravelAction(kind="step", message="rock"))

    with pytest.raises(ValueError, match="ago must be > 0"):
        env.step(TimetravelAction(kind="branch", instruction="invalid", ago=0))


def test_branch_with_ago_too_large_is_rejected() -> None:
    env = TimetravelEnvironment(max_rounds=5)
    env.reset()
    env.step(TimetravelAction(kind="step", message="paper"))

    with pytest.raises(ValueError, match="cannot exceed current_step"):
        env.step(TimetravelAction(kind="branch", instruction="invalid", ago=2))


def test_branch_stops_old_timeline_and_continues_from_new_one() -> None:
    env = TimetravelEnvironment(max_rounds=5)
    reset_obs = env.reset()
    initial_timeline = reset_obs.active_timeline_id

    env.step(TimetravelAction(kind="step", message="rock"))
    env.step(TimetravelAction(kind="step", message="paper"))
    branched = env.step(TimetravelAction(kind="branch", instruction="try a different move", ago=1))

    assert branched.active_timeline_id != initial_timeline
    assert env._timelines[initial_timeline].status == "abandoned"


def test_abandon_updates_timeline_status_and_done() -> None:
    env = TimetravelEnvironment(max_rounds=5)
    env.reset()

    obs = env.step(TimetravelAction(kind="abandon"))

    assert obs.done is True
    assert obs.timeline_status == "abandoned"


def test_reset_clears_timeline_and_metadata() -> None:
    env = TimetravelEnvironment(max_rounds=5)
    env.reset()
    env.step(TimetravelAction(kind="step", message="scissors"))
    env.step(TimetravelAction(kind="abandon"))

    reset_obs = env.reset()

    assert reset_obs.current_step == 0
    assert reset_obs.event_log_size == 1
    assert reset_obs.last_branch_event is None
    assert len(reset_obs.metadata.get("event_log", [])) == 1
