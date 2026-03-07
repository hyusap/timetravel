# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Timetravel Environment."""

from typing import Any, Dict, Literal, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation


class TimetravelAction(Action):
    """Action for the reverse-only timetravel environment."""

    kind: Literal["step", "branch", "abandon", "pause"] = Field(
        default="step",
        description="Temporal control action kind",
    )
    message: str = Field(default="", description="Message for normal step action")
    instruction: str = Field(
        default="",
        description="Instruction attached to a branch action",
    )
    ago: Optional[int] = Field(
        default=None,
        description="Number of steps to rewind for branch actions",
    )


class TimetravelObservation(Observation):
    """Observation from the reverse-only timetravel environment."""

    echoed_message: str = Field(default="", description="The echoed message")
    message_length: int = Field(default=0, description="Length of the echoed message")
    current_step: int = Field(default=0, description="Current active timeline step count")
    active_timeline_id: str = Field(default="", description="Current active timeline ID")
    timeline_status: Literal["active", "paused", "abandoned", "done"] = Field(
        default="active",
        description="Status of the active timeline",
    )
    last_branch_event: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata for the most recent branch action",
    )
    event_log_size: int = Field(default=0, description="Total number of logged events")
