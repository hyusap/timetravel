from __future__ import annotations

import random
from typing import Any

from .temporal_single_timeline import TemporalSingleTimelineEnv

_ALLOWED_MOVES = ("rock", "paper", "scissors")


class TimetravelEnvironment(TemporalSingleTimelineEnv):
    """Rock-Paper-Scissors environment built on temporal-control base class."""

    def __init__(self, max_rounds: int = 5) -> None:
        super().__init__()
        self._max_rounds = max_rounds

    def _initial_domain_state(self) -> dict[str, Any]:
        return {
            "game": "rock_paper_scissors",
            "max_rounds": self._max_rounds,
            "round_index": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "last_agent_move": None,
            "last_opponent_move": None,
            "last_outcome": None,
            "score": 0,
        }

    def _apply_domain_step(self, message: str) -> tuple[float, bool, str]:
        if message not in _ALLOWED_MOVES:
            raise ValueError("RPS move must be one of: rock, paper, scissors")

        opponent_move = random.choice(_ALLOWED_MOVES)
        outcome = self._resolve_outcome(agent_move=message, opponent_move=opponent_move)

        reward = 0.0
        if outcome == "win":
            reward = 1.0
            self._domain_state["wins"] += 1
        elif outcome == "loss":
            reward = -1.0
            self._domain_state["losses"] += 1
        else:
            self._domain_state["draws"] += 1

        self._domain_state["score"] += int(reward)
        self._domain_state["round_index"] += 1
        self._domain_state["last_agent_move"] = message
        self._domain_state["last_opponent_move"] = opponent_move
        self._domain_state["last_outcome"] = outcome

        done = self._domain_state["round_index"] >= self._domain_state["max_rounds"]
        summary = f"Round {self._domain_state['round_index']}: {message} vs {opponent_move} -> {outcome}"
        return reward, done, summary

    def _build_domain_view(self) -> dict[str, Any]:
        return {
            "game": self._domain_state["game"],
            "max_rounds": self._domain_state["max_rounds"],
            "round_index": self._domain_state["round_index"],
            "wins": self._domain_state["wins"],
            "losses": self._domain_state["losses"],
            "draws": self._domain_state["draws"],
            "score": self._domain_state["score"],
            "last_agent_move": self._domain_state["last_agent_move"],
            "last_opponent_move": self._domain_state["last_opponent_move"],
            "last_outcome": self._domain_state["last_outcome"],
        }

    def _resolve_outcome(self, agent_move: str, opponent_move: str) -> str:
        if agent_move == opponent_move:
            return "draw"

        wins_against = {
            "rock": "scissors",
            "paper": "rock",
            "scissors": "paper",
        }
        if wins_against[agent_move] == opponent_move:
            return "win"
        return "loss"
