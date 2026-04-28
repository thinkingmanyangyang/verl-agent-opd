from __future__ import annotations

from pathlib import Path
from typing import Any

from agent_system.skill_opd.config import load_export_config
from agent_system.skill_opd.rollout_exporter import RolloutExporter


_INITIALIZED_PATHS: set[str] = set()


def maybe_export_rollout(
    config: Any,
    tokenizer: Any,
    total_batch_list: list[list[dict[str, Any]]],
    total_infos: list[list[dict[str, Any]]],
    episode_rewards: Any,
    episode_lengths: Any,
    success: dict[str, Any],
    traj_uid: Any,
    tool_callings: Any,
) -> None:
    """Export rollout data when `+skill_opd.export_rollouts=True` is set."""

    export_config = load_export_config(config)
    if not export_config.export_rollouts:
        return

    export_path = str(Path(export_config.export_path))
    overwrite = False
    if export_config.overwrite and export_path not in _INITIALIZED_PATHS:
        overwrite = True
        _INITIALIZED_PATHS.add(export_path)

    count = RolloutExporter(
        tokenizer=tokenizer,
        config=config,
        export_config=export_config,
    ).export(
        total_batch_list=total_batch_list,
        total_infos=total_infos,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        success=success,
        traj_uid=traj_uid,
        tool_callings=tool_callings,
        overwrite=overwrite,
    )

    print(f"[SkillOPD] exported {count} rollout trajectories to {export_path}")

