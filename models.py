from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from openenv.core.env_server.types import Action, Observation


class TimetravelAction(Action):
    """Temporal-control action schema for all current and future environments."""

    kind: Literal["step", "branch", "abandon", "pause"] = Field(default="step")
    message: str | None = Field(default=None, description="Domain step payload (RPS move for now)")
    instruction: str | None = Field(default=None, description="Instruction to attach when branching")
    ago: int | None = Field(default=None, description="How many steps back to rewind on branch")

    @model_validator(mode="after")
    def _validate(self) -> "TimetravelAction":
        if self.kind == "step" and not (self.message and self.message.strip()):
            raise ValueError("step actions require a non-empty message")
        if self.kind == "branch":
            if self.ago is None:
                raise ValueError("branch actions require ago")
            if not (self.instruction and self.instruction.strip()):
                raise ValueError("branch actions require a non-empty instruction")
        return self


class TimetravelObservation(Observation):
    """Generic observation schema with temporal metadata and domain world-state."""

    summary: str = Field(default="", description="Short human-readable transition summary")
    world_state: dict[str, Any] = Field(default_factory=dict, description="Domain-specific world state")

    current_step: int = Field(default=0, description="Current active step index")
    active_timeline_id: str = Field(default="", description="Current active timeline id")
    last_branch_event: str | None = Field(default=None, description="Most recent branch event id")
    timeline_status: Literal["active", "paused", "abandoned", "done"] = Field(default="active")
    event_log_size: int = Field(default=0, description="Number of events captured in metadata event log")
