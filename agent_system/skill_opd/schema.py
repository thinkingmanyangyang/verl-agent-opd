from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RolloutStepRecord:
    """One active environment step from a student rollout."""

    step_id: int
    prompt_text: str | None
    response_text: str | None
    reward: float | None
    done: bool | None
    active_mask: bool
    is_action_valid: bool | None
    response_token_ids: list[int] | None = None
    uid: str | None = None
    traj_uid: str | None = None
    data_source: str | None = None
    info: dict[str, Any] = field(default_factory=dict)
    available_keys: list[str] = field(default_factory=list)


@dataclass
class RolloutTrajectoryRecord:
    """A full student trajectory ready for offline teacher scoring."""

    trajectory_id: str
    env_name: str | None
    total_reward: float | None
    episode_length: int | None
    success: bool | None
    tool_call_count: int | None
    steps: list[RolloutStepRecord]
    success_metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

