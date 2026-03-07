# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Timetravel Environment Implementation."""

from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:  # pragma: no cover - supports both package and local execution
    from ..models import TimetravelAction, TimetravelObservation
except ImportError:  # pragma: no cover
    from models import TimetravelAction, TimetravelObservation


class TimetravelEnvironment(Environment):
    """Reverse-only temporal environment with branch and abandon controls."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the timetravel environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._timeline_counter = 0
        self._event_counter = 0
        self._active_timeline_id = ""
        self._timelines: Dict[str, Dict[str, Any]] = {}
        self._meta_events: List[Dict[str, Any]] = []
        self._last_branch_event: Optional[Dict[str, Any]] = None
        self._episode_done = False
        self._initialize_temporal_state()

    def _initialize_temporal_state(self) -> None:
        """Initialize timeline structures for a fresh episode."""
        self._timeline_counter = 1
        self._event_counter = 0
        self._active_timeline_id = f"timeline-{self._timeline_counter}"
        self._timelines = {
            self._active_timeline_id: {
                "timeline_id": self._active_timeline_id,
                "status": "active",
                "source_timeline_id": None,
                "steps": [],
                "archived_future": [],
            }
        }
        self._meta_events = []
        self._last_branch_event = None
        self._episode_done = False
        self._sync_step_count()

    def _active_timeline(self) -> Dict[str, Any]:
        """Return mutable data for the current active timeline."""
        return self._timelines[self._active_timeline_id]

    def _sync_step_count(self) -> None:
        """Keep OpenEnv state.step_count aligned with active timeline steps."""
        self._state.step_count = len(self._active_timeline()["steps"])

    def _next_event_id(self) -> str:
        """Allocate a monotonic event identifier."""
        self._event_counter += 1
        return f"event-{self._event_counter}"

    def _log_event(
        self,
        *,
        event_type: str,
        message: str = "",
        instruction: str = "",
        ago: Optional[int] = None,
        reward: float = 0.0,
        done: bool = False,
        source_timeline_id: Optional[str] = None,
        parent_event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a structured meta-trajectory event."""
        event = {
            "event_id": self._next_event_id(),
            "event_type": event_type,
            "step_index": self._state.step_count,
            "parent_event_id": parent_event_id,
            "timeline_id": self._active_timeline_id,
            "source_timeline_id": source_timeline_id,
            "ago": ago,
            "instruction": instruction,
            "message": message,
            "reward": reward,
            "done": done,
            "status_after_event": self._active_timeline()["status"],
        }
        self._meta_events.append(event)
        return event

    def _build_observation(
        self,
        *,
        echoed_message: str,
        reward: float,
        done: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TimetravelObservation:
        """Build a consistent observation payload with temporal metadata."""
        self._sync_step_count()
        merged_metadata = {
            "current_step": self._state.step_count,
            "active_timeline_id": self._active_timeline_id,
            "timeline_status": self._active_timeline()["status"],
            "last_branch_event": self._last_branch_event,
            "event_log_size": len(self._meta_events),
        }
        if metadata:
            merged_metadata.update(metadata)

        return TimetravelObservation(
            echoed_message=echoed_message,
            message_length=len(echoed_message),
            current_step=self._state.step_count,
            active_timeline_id=self._active_timeline_id,
            timeline_status=self._active_timeline()["status"],
            last_branch_event=self._last_branch_event,
            event_log_size=len(self._meta_events),
            done=done,
            reward=reward,
            metadata=merged_metadata,
        )

    def reset(self) -> TimetravelObservation:
        """
        Reset the environment.

        Returns:
            TimetravelObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._initialize_temporal_state()

        return self._build_observation(
            echoed_message="Timetravel environment ready!",
            reward=0.0,
            done=False,
            metadata={"event_type": "reset", "reset_count": self._reset_count},
        )

    def step(self, action: TimetravelAction) -> TimetravelObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: TimetravelAction containing the message to echo

        Returns:
            TimetravelObservation with the echoed message and its length
        """
        if self._episode_done:
            return self._build_observation(
                echoed_message="Episode is done. Call reset() to start a new trajectory.",
                reward=0.0,
                done=True,
                metadata={"event_type": "terminal_noop"},
            )

        kind = action.kind or "step"
        if kind == "branch":
            return self._handle_branch(action)
        if kind in {"abandon", "pause"}:
            return self._handle_abandon(kind)
        return self._handle_step(action)

    def _handle_step(self, action: TimetravelAction) -> TimetravelObservation:
        """Handle normal step actions on the active timeline."""
        message = action.message
        reward = len(message) * 0.1

        parent_event_id = self._active_timeline()["steps"][-1]["event_id"] if self._active_timeline()["steps"] else None
        event = self._log_event(
            event_type="step",
            message=message,
            reward=reward,
            done=False,
            parent_event_id=parent_event_id,
        )

        self._active_timeline()["steps"].append(
            {
                "event_id": event["event_id"],
                "message": message,
                "reward": reward,
            }
        )
        self._sync_step_count()

        return self._build_observation(
            echoed_message=message,
            reward=reward,
            done=False,
            metadata={
                "event_type": "step",
                "event_id": event["event_id"],
                "original_message": message,
            },
        )

    def _handle_branch(self, action: TimetravelAction) -> TimetravelObservation:
        """Handle reverse-only branch actions with strict ago>0 semantics."""
        ago = action.ago
        if ago is None:
            raise ValueError("branch action requires 'ago'")
        if ago <= 0:
            raise ValueError("branch requires ago > 0")
        if ago > self._state.step_count:
            raise ValueError("branch ago cannot exceed current_step")

        old_timeline_id = self._active_timeline_id
        old_timeline = self._active_timeline()
        old_steps = old_timeline["steps"]
        rewind_index = self._state.step_count - ago

        prefix_steps = deepcopy(old_steps[:rewind_index])
        archived_future = deepcopy(old_steps[rewind_index:])
        parent_event_id = prefix_steps[-1]["event_id"] if prefix_steps else None

        old_timeline["status"] = "paused"
        old_timeline["archived_future"].append(
            {
                "from_step": rewind_index,
                "events": archived_future,
            }
        )

        self._timeline_counter += 1
        new_timeline_id = f"timeline-{self._timeline_counter}"
        self._timelines[new_timeline_id] = {
            "timeline_id": new_timeline_id,
            "status": "active",
            "source_timeline_id": old_timeline_id,
            "steps": prefix_steps,
            "archived_future": [],
        }
        self._active_timeline_id = new_timeline_id
        self._sync_step_count()

        event = self._log_event(
            event_type="branch",
            instruction=action.instruction,
            ago=ago,
            reward=0.0,
            done=False,
            source_timeline_id=old_timeline_id,
            parent_event_id=parent_event_id,
        )
        self._last_branch_event = {
            "event_id": event["event_id"],
            "from_timeline_id": old_timeline_id,
            "to_timeline_id": new_timeline_id,
            "source_step": self._state.step_count + ago,
            "rewind_to_step": rewind_index,
            "ago": ago,
            "instruction": action.instruction,
        }

        return self._build_observation(
            echoed_message=f"Branched back {ago} step(s): {action.instruction}",
            reward=0.0,
            done=False,
            metadata={
                "event_type": "branch",
                "event_id": event["event_id"],
                "source_timeline_id": old_timeline_id,
                "archived_future_steps": len(archived_future),
            },
        )

    def _handle_abandon(self, kind: str) -> TimetravelObservation:
        """Handle abandon/pause by ending the current active timeline."""
        self._active_timeline()["status"] = "abandoned"
        self._episode_done = True

        parent_event_id = self._active_timeline()["steps"][-1]["event_id"] if self._active_timeline()["steps"] else None
        event = self._log_event(
            event_type=kind,
            reward=0.0,
            done=True,
            parent_event_id=parent_event_id,
        )
        return self._build_observation(
            echoed_message="Active timeline abandoned.",
            reward=0.0,
            done=True,
            metadata={
                "event_type": kind,
                "event_id": event["event_id"],
            },
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
