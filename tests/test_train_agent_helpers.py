from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.reverse_code_door import TemporalAction
from train.reverse_code_door_agent import parse_action


def test_parse_unlock_and_branch_actions() -> None:
    unlock = parse_action("ACTION: unlock 042")
    assert unlock == TemporalAction(command="unlock", unlock_code="042")

    branch = parse_action("ACTION: branch 3 use code 042 at door")
    assert branch == TemporalAction(kind="branch", ago=3, instruction="use code 042 at door")


def test_parse_invalid_returns_none() -> None:
    assert parse_action("abandon") is None
    assert parse_action("branch x nope") is None
    assert parse_action("nonsense") is None
    assert parse_action("move forward to oracle then inspect") is None


def test_parse_multiline_reasoning_uses_action_line() -> None:
    text = """<think>
    I should go toward the oracle first.
    </think>
    ACTION: forward
    """
    action = parse_action(text)
    assert action == TemporalAction(command="forward")

