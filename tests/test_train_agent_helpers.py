from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.reverse_code_door import TemporalAction
from train.reverse_code_door_agent import format_action, parse_action


def test_parse_unlock_and_branch_actions() -> None:
    unlock = parse_action('{"command":"unlock","code":"042"}')
    assert unlock == TemporalAction(command="unlock", unlock_code="042")

    branch = parse_action('{"command":"branch","ago":3,"instruction":"use code 042 at door"}')
    assert branch == TemporalAction(kind="branch", ago=3, instruction="use code 042 at door")


def test_parse_invalid_returns_none() -> None:
    assert parse_action("abandon") is None
    assert parse_action("branch x nope") is None
    assert parse_action("nonsense") is None


def test_parse_json_from_text_wrapper() -> None:
    text = "Model output: {\"command\":\"forward\"}"
    action = parse_action(text)
    assert action == TemporalAction(command="forward")


def test_format_action_json() -> None:
    assert format_action(TemporalAction(command="forward")) == '{"command":"forward"}'
