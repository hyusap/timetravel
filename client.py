# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Timetravel Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import TimetravelAction, TimetravelObservation


class TimetravelEnv(
    EnvClient[TimetravelAction, TimetravelObservation, State]
):
    """
    Client for the Timetravel Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TimetravelEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(TimetravelAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TimetravelEnv.from_docker_image("timetravel-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TimetravelAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TimetravelAction) -> Dict:
        """
        Convert TimetravelAction to JSON payload for step message.

        Args:
            action: TimetravelAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload: Dict[str, object] = {
            "kind": action.kind,
            "message": action.message,
        }
        if action.instruction:
            payload["instruction"] = action.instruction
        if action.ago is not None:
            payload["ago"] = action.ago
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[TimetravelObservation]:
        """
        Parse server response into StepResult[TimetravelObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TimetravelObservation
        """
        obs_data = payload.get("observation", {})
        observation = TimetravelObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            current_step=obs_data.get("current_step", 0),
            active_timeline_id=obs_data.get("active_timeline_id", ""),
            timeline_status=obs_data.get("timeline_status", "active"),
            last_branch_event=obs_data.get("last_branch_event"),
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

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

    def branch(self, instruction: str, ago: int):
        """Convenience helper for single-instruction rewind branching."""
        return self.step(
            TimetravelAction(
                kind="branch",
                instruction=instruction,
                ago=ago,
            )
        )

    def abandon(self):
        """Convenience helper to abandon the active timeline."""
        return self.step(TimetravelAction(kind="abandon"))

    def pause(self):
        """Alias for abandon in reverse-only v0."""
        return self.step(TimetravelAction(kind="pause"))
