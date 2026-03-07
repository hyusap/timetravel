from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import TimetravelAction
from server.timetravel_environment import TimetravelEnvironment


def test_branch_rewinds_and_continues_on_forked_timeline_only():
    env = TimetravelEnvironment()
    env.reset()

    env.step(TimetravelAction(message="one"))
    env.step(TimetravelAction(message="two"))
    env.step(TimetravelAction(message="three"))

    branch_obs = env.step(
        TimetravelAction(kind="branch", instruction="Avoid prior path", ago=2)
    )

    assert branch_obs.current_step == 1
    assert branch_obs.done is False
    assert branch_obs.last_branch_event is not None
    assert branch_obs.last_branch_event["ago"] == 2

    old_timeline_id = branch_obs.last_branch_event["from_timeline_id"]
    new_timeline_id = branch_obs.last_branch_event["to_timeline_id"]

    assert old_timeline_id != new_timeline_id
    assert env._timelines[old_timeline_id]["status"] == "paused"
    assert env._timelines[new_timeline_id]["status"] == "active"

    step_obs = env.step(TimetravelAction(message="new-two"))
    assert step_obs.current_step == 2
    assert step_obs.echoed_message == "new-two"


def test_branch_rejects_ago_zero():
    env = TimetravelEnvironment()
    env.reset()
    env.step(TimetravelAction(message="one"))

    with pytest.raises(ValueError, match="ago > 0"):
        env.step(TimetravelAction(kind="branch", instruction="retry", ago=0))


def test_branch_rejects_ago_greater_than_step_count():
    env = TimetravelEnvironment()
    env.reset()
    env.step(TimetravelAction(message="one"))

    with pytest.raises(ValueError, match="cannot exceed"):
        env.step(TimetravelAction(kind="branch", instruction="retry", ago=2))


def test_abandon_sets_done_and_abandoned_status():
    env = TimetravelEnvironment()
    env.reset()
    env.step(TimetravelAction(message="one"))

    obs = env.step(TimetravelAction(kind="abandon"))

    assert obs.done is True
    assert obs.timeline_status == "abandoned"


def test_reset_clears_temporal_state():
    env = TimetravelEnvironment()
    env.reset()
    env.step(TimetravelAction(message="one"))
    env.step(TimetravelAction(kind="branch", instruction="retry", ago=1))

    obs = env.reset()

    assert obs.current_step == 0
    assert obs.last_branch_event is None
    assert obs.event_log_size == 0
    assert obs.timeline_status == "active"


def test_backward_compatible_step_action_without_kind():
    env = TimetravelEnvironment()
    env.reset()

    obs = env.step(TimetravelAction(message="hello"))

    assert obs.echoed_message == "hello"
    assert obs.current_step == 1
    assert obs.done is False
