from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from timetravel.models import TimetravelAction, TimetravelObservation

TimelineStatus = Literal["active", "paused", "abandoned", "done"]


@dataclass
class TemporalEvent:
    event_id: str
    event_type: str
    step_index: int
    parent_event_id: str | None
    timeline_id: str
    source_timeline_id: str | None
    ago: int | None
    instruction: str | None
    message: str | None
    reward: float
    done: bool
    status_after_event: TimelineStatus
    snapshot: dict[str, Any]


@dataclass
class TimelineRecord:
    timeline_id: str
    status: TimelineStatus = "active"
    event_ids: list[str] = field(default_factory=list)
    archived_future_event_ids: list[str] = field(default_factory=list)


class TemporalSingleTimelineEnv(Environment, ABC):
    """Reusable temporal-control environment with one active timeline."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._event_counter = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._events: dict[str, TemporalEvent] = {}
        self._event_order: list[str] = []
        self._timelines: dict[str, TimelineRecord] = {}
        self._active_timeline_id: str = ""
        self._domain_state: dict[str, Any] = {}
        self._last_branch_event: str | None = None

    def reset(self) -> TimetravelObservation:
        self._event_counter = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._events = {}
        self._event_order = []
        self._timelines = {}
        self._last_branch_event = None

        timeline_id = self._new_timeline_id()
        self._active_timeline_id = timeline_id
        self._timelines[timeline_id] = TimelineRecord(timeline_id=timeline_id, status="active")

        self._domain_state = self._initial_domain_state()

        self._record_event(
            event_type="reset",
            timeline_id=timeline_id,
            parent_event_id=None,
            source_timeline_id=None,
            ago=None,
            instruction=None,
            message=None,
            reward=0.0,
            done=False,
            step_index=0,
        )
        return self._make_observation(reward=0.0, done=False, summary="Environment reset")

    def step(self, action: TimetravelAction) -> TimetravelObservation:  # type: ignore[override]
        if action.kind == "step":
            return self._handle_domain_step(action)
        if action.kind == "branch":
            return self._handle_branch(action)
        if action.kind in {"abandon", "pause"}:
            return self._handle_abandon(action.kind)
        raise ValueError(f"Unsupported action kind: {action.kind}")

    def _handle_domain_step(self, action: TimetravelAction) -> TimetravelObservation:
        message = (action.message or "").strip().lower()
        if not message:
            raise ValueError("step action requires non-empty message")

        reward, done, summary = self._apply_domain_step(message)
        self._state.step_count += 1

        timeline = self._timelines[self._active_timeline_id]
        if done:
            timeline.status = "done"

        self._record_event(
            event_type="step",
            timeline_id=self._active_timeline_id,
            parent_event_id=self._last_event_id(self._active_timeline_id),
            source_timeline_id=None,
            ago=None,
            instruction=None,
            message=message,
            reward=reward,
            done=done,
            step_index=self._state.step_count,
        )

        return self._make_observation(reward=reward, done=done, summary=summary)

    def _handle_branch(self, action: TimetravelAction) -> TimetravelObservation:
        ago = action.ago
        instruction = (action.instruction or "").strip()
        if ago is None:
            raise ValueError("branch action requires ago")
        if ago <= 0:
            raise ValueError("branch ago must be > 0")
        if ago > self._state.step_count:
            raise ValueError("branch ago cannot exceed current_step")
        if not instruction:
            raise ValueError("branch action requires non-empty instruction")

        old_timeline_id = self._active_timeline_id
        old_timeline = self._timelines[old_timeline_id]

        target_step = self._state.step_count - ago
        rewind_idx = None
        for idx, event_id in enumerate(old_timeline.event_ids):
            if self._events[event_id].step_index == target_step:
                rewind_idx = idx
        if rewind_idx is None:
            raise ValueError("Unable to resolve rewind target")

        rewind_event_id = old_timeline.event_ids[rewind_idx]
        rewind_event = self._events[rewind_event_id]

        archived_future = old_timeline.event_ids[rewind_idx + 1 :]
        old_timeline.archived_future_event_ids.extend(archived_future)
        old_timeline.status = "abandoned"

        new_timeline_id = self._new_timeline_id()
        new_timeline = TimelineRecord(timeline_id=new_timeline_id, status="active")
        new_timeline.event_ids = old_timeline.event_ids[: rewind_idx + 1]
        self._timelines[new_timeline_id] = new_timeline

        self._domain_state = deepcopy(rewind_event.snapshot)
        self._state.step_count = target_step
        self._active_timeline_id = new_timeline_id

        branch_event_id = self._record_event(
            event_type="branch",
            timeline_id=new_timeline_id,
            parent_event_id=rewind_event_id,
            source_timeline_id=old_timeline_id,
            ago=ago,
            instruction=instruction,
            message=None,
            reward=0.0,
            done=False,
            step_index=self._state.step_count,
        )
        self._last_branch_event = branch_event_id

        return self._make_observation(
            reward=0.0,
            done=False,
            summary=f"Branched {ago} step(s) back with instruction: {instruction}",
        )

    def _handle_abandon(self, event_type: str) -> TimetravelObservation:
        self._timelines[self._active_timeline_id].status = "abandoned"
        self._record_event(
            event_type=event_type,
            timeline_id=self._active_timeline_id,
            parent_event_id=self._last_event_id(self._active_timeline_id),
            source_timeline_id=None,
            ago=None,
            instruction=None,
            message=None,
            reward=0.0,
            done=True,
            step_index=self._state.step_count,
        )
        return self._make_observation(reward=0.0, done=True, summary="Timeline abandoned")

    def _record_event(
        self,
        *,
        event_type: str,
        timeline_id: str,
        parent_event_id: str | None,
        source_timeline_id: str | None,
        ago: int | None,
        instruction: str | None,
        message: str | None,
        reward: float,
        done: bool,
        step_index: int,
    ) -> str:
        self._event_counter += 1
        event_id = f"evt-{self._event_counter}"
        status = self._timelines[timeline_id].status

        event = TemporalEvent(
            event_id=event_id,
            event_type=event_type,
            step_index=step_index,
            parent_event_id=parent_event_id,
            timeline_id=timeline_id,
            source_timeline_id=source_timeline_id,
            ago=ago,
            instruction=instruction,
            message=message,
            reward=reward,
            done=done,
            status_after_event=status,
            snapshot=deepcopy(self._domain_state),
        )
        self._events[event_id] = event
        self._event_order.append(event_id)
        self._timelines[timeline_id].event_ids.append(event_id)
        return event_id

    def _last_event_id(self, timeline_id: str) -> str | None:
        event_ids = self._timelines[timeline_id].event_ids
        if not event_ids:
            return None
        return event_ids[-1]

    def _new_timeline_id(self) -> str:
        return f"tl-{uuid4().hex[:8]}"

    def _make_observation(self, *, reward: float, done: bool, summary: str) -> TimetravelObservation:
        timeline = self._timelines[self._active_timeline_id]
        domain_view = self._build_domain_view()

        metadata = {
            "current_step": self._state.step_count,
            "active_timeline_id": self._active_timeline_id,
            "last_branch_event": self._last_branch_event,
            "timeline_status": timeline.status,
            "event_log": [self._event_to_metadata(self._events[event_id]) for event_id in self._event_order],
        }

        return TimetravelObservation(
            summary=summary,
            world_state=domain_view,
            current_step=self._state.step_count,
            active_timeline_id=self._active_timeline_id,
            last_branch_event=self._last_branch_event,
            timeline_status=timeline.status,
            event_log_size=len(self._event_order),
            reward=reward,
            done=done,
            metadata=metadata,
        )

    def _event_to_metadata(self, event: TemporalEvent) -> dict[str, Any]:
        return {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "step_index": event.step_index,
            "parent_event_id": event.parent_event_id,
            "timeline_id": event.timeline_id,
            "source_timeline_id": event.source_timeline_id,
            "ago": event.ago,
            "instruction": event.instruction,
            "message": event.message,
            "reward": event.reward,
            "done": event.done,
            "status_after_event": event.status_after_event,
        }

    @property
    def state(self) -> State:
        return self._state

    @abstractmethod
    def _initial_domain_state(self) -> dict[str, Any]:
        """Return initial per-episode domain state."""

    @abstractmethod
    def _apply_domain_step(self, message: str) -> tuple[float, bool, str]:
        """Apply a normal step and return (reward, done, summary)."""

    @abstractmethod
    def _build_domain_view(self) -> dict[str, Any]:
        """Build serializable world state for observations."""
