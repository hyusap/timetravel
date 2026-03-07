from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TimetravelAction, TimetravelObservation


class TimetravelEnv(EnvClient[TimetravelAction, TimetravelObservation, State]):
    """Client for temporal-control environments."""

    def send_step(self, message: str) -> StepResult[TimetravelObservation]:
        return self.step(TimetravelAction(kind="step", message=message))

    def branch(self, instruction: str, ago: int) -> StepResult[TimetravelObservation]:
        return self.step(TimetravelAction(kind="branch", instruction=instruction, ago=ago))

    def abandon(self) -> StepResult[TimetravelObservation]:
        return self.step(TimetravelAction(kind="abandon"))

    def _step_payload(self, action: TimetravelAction) -> dict[str, Any]:
        return {
            "kind": action.kind,
            "message": action.message,
            "instruction": action.instruction,
            "ago": action.ago,
        }

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[TimetravelObservation]:
        obs_data = payload.get("observation", {})
        observation = TimetravelObservation(
            summary=obs_data.get("summary", ""),
            world_state=obs_data.get("world_state", {}),
            current_step=obs_data.get("current_step", 0),
            active_timeline_id=obs_data.get("active_timeline_id", ""),
            last_branch_event=obs_data.get("last_branch_event"),
            timeline_status=obs_data.get("timeline_status", "active"),
            event_log_size=obs_data.get("event_log_size", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
