from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SkillOPDExportConfig:
    """Config values for offline rollout export."""

    export_rollouts: bool = False
    export_path: str = "outputs/skill_opd/rollouts/rollouts.jsonl"
    include_text: bool = True
    include_token_ids: bool = True
    include_infos: bool = True
    overwrite: bool = False


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict/OmegaConf-like/attribute config objects."""

    if config is None:
        return default

    if isinstance(config, dict):
        return config.get(key, default)

    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass

    return getattr(config, key, default)


def load_export_config(root_config: Any) -> SkillOPDExportConfig:
    """Build export config from `config.skill_opd` if it exists."""

    skill_cfg = cfg_get(root_config, "skill_opd", None)
    if skill_cfg is None:
        return SkillOPDExportConfig()

    return SkillOPDExportConfig(
        export_rollouts=bool(cfg_get(skill_cfg, "export_rollouts", False)),
        export_path=str(
            cfg_get(skill_cfg, "export_path", "outputs/skill_opd/rollouts/rollouts.jsonl")
        ),
        include_text=bool(cfg_get(skill_cfg, "include_text", True)),
        include_token_ids=bool(cfg_get(skill_cfg, "include_token_ids", True)),
        include_infos=bool(cfg_get(skill_cfg, "include_infos", True)),
        overwrite=bool(cfg_get(skill_cfg, "overwrite", False)),
    )

