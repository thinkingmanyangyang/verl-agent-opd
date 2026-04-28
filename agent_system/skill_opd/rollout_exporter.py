from __future__ import annotations

import json
from typing import Any

from agent_system.skill_opd.config import SkillOPDExportConfig, cfg_get
from agent_system.skill_opd.io import JsonlWriter, to_jsonable
from agent_system.skill_opd.schema import RolloutStepRecord, RolloutTrajectoryRecord


EXPORT_SCHEMA_VERSION = "skill_opd.rollout.v1"


def _plain(value: Any) -> Any:
    return to_jsonable(value)


def _as_float(value: Any) -> float | None:
    value = _plain(value)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return _as_float(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    value = _plain(value)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return _as_int(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool | None:
    value = _plain(value)
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return _as_bool(value[0])
    if isinstance(value, str):
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
    return bool(value)


def _as_token_list(value: Any) -> list[int] | None:
    value = _plain(value)
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if not isinstance(value, list):
        return None
    if value and isinstance(value[0], list):
        value = value[0]
    tokens: list[int] = []
    for token in value:
        try:
            tokens.append(int(token))
        except (TypeError, ValueError):
            return None
    return tokens


def _get_first(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _format_raw_prompt(raw_prompt: Any) -> str | None:
    raw_prompt = _plain(raw_prompt)
    if raw_prompt is None:
        return None
    if isinstance(raw_prompt, str):
        return raw_prompt
    if isinstance(raw_prompt, list):
        parts: list[str] = []
        for message in raw_prompt:
            if isinstance(message, dict):
                role = message.get("role", "message")
                content = message.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(message))
        return "\n".join(parts)
    return json.dumps(raw_prompt, ensure_ascii=False)


def _decode_tokens(tokenizer: Any, token_ids: list[int] | None) -> str | None:
    if not token_ids:
        return None
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception:
        return None


def _decode_prompt(data: dict[str, Any], tokenizer: Any) -> str | None:
    raw_prompt = _get_first(data, ["raw_prompt", "chat", "prompt"])
    prompt_text = _format_raw_prompt(raw_prompt)
    if prompt_text:
        return prompt_text

    prompt_tokens = _as_token_list(
        _get_first(data, ["prompts", "raw_prompt_ids", "input_ids"])
    )
    return _decode_tokens(tokenizer, prompt_tokens)


def _decode_response(data: dict[str, Any], tokenizer: Any) -> tuple[str | None, list[int] | None]:
    response_text = _get_first(data, ["response_text", "action_text", "text_action"])
    if response_text is not None:
        return str(_plain(response_text)), _as_token_list(_get_first(data, ["responses"]))

    response_token_ids = _as_token_list(_get_first(data, ["responses"]))
    return _decode_tokens(tokenizer, response_token_ids), response_token_ids


def _value_at_index(value: Any, index: int) -> Any:
    value = _plain(value)
    if isinstance(value, list):
        if index < len(value):
            return value[index]
        return None
    return value


class RolloutExporter:
    """Convert verl-agent rollout buffers into JSONL trajectory records."""

    def __init__(
        self,
        tokenizer: Any,
        config: Any,
        export_config: SkillOPDExportConfig,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.export_config = export_config

    def build_records(
        self,
        total_batch_list: list[list[dict[str, Any]]],
        total_infos: list[list[dict[str, Any]]],
        episode_rewards: Any,
        episode_lengths: Any,
        success: dict[str, Any],
        traj_uid: Any,
        tool_callings: Any,
    ) -> list[RolloutTrajectoryRecord]:
        records: list[RolloutTrajectoryRecord] = []
        env_name = cfg_get(cfg_get(self.config, "env", None), "env_name", None)

        for traj_index, step_items in enumerate(total_batch_list):
            active_indices = [
                step_index
                for step_index, step_data in enumerate(step_items)
                if _as_bool(step_data.get("active_masks", True)) is not False
            ]
            last_active_index = active_indices[-1] if active_indices else None
            steps: list[RolloutStepRecord] = []

            for step_index in active_indices:
                step_data = step_items[step_index]
                step_info = {}
                if self.export_config.include_infos and traj_index < len(total_infos):
                    if step_index < len(total_infos[traj_index]):
                        step_info = _plain(total_infos[traj_index][step_index]) or {}

                prompt_text = None
                response_text = None
                response_token_ids = None
                if self.export_config.include_text:
                    prompt_text = _decode_prompt(step_data, self.tokenizer)
                    response_text, response_token_ids = _decode_response(step_data, self.tokenizer)
                elif self.export_config.include_token_ids:
                    response_token_ids = _as_token_list(_get_first(step_data, ["responses"]))

                if not self.export_config.include_token_ids:
                    response_token_ids = None

                steps.append(
                    RolloutStepRecord(
                        step_id=step_index,
                        prompt_text=prompt_text,
                        response_text=response_text,
                        response_token_ids=response_token_ids,
                        reward=_as_float(step_data.get("rewards")),
                        done=step_index == last_active_index if last_active_index is not None else None,
                        active_mask=True,
                        is_action_valid=_as_bool(step_data.get("is_action_valid")),
                        uid=str(_plain(step_data.get("uid"))) if step_data.get("uid") is not None else None,
                        traj_uid=(
                            str(_plain(step_data.get("traj_uid")))
                            if step_data.get("traj_uid") is not None
                            else None
                        ),
                        data_source=(
                            str(_plain(step_data.get("data_source")))
                            if step_data.get("data_source") is not None
                            else None
                        ),
                        info=step_info,
                        available_keys=sorted(str(key) for key in step_data.keys()),
                    )
                )

            trajectory_id = str(_value_at_index(traj_uid, traj_index) or f"traj-{traj_index}")
            success_metrics = self._success_metrics_for_index(success, traj_index)
            records.append(
                RolloutTrajectoryRecord(
                    trajectory_id=trajectory_id,
                    env_name=str(env_name) if env_name is not None else None,
                    total_reward=_as_float(_value_at_index(episode_rewards, traj_index)),
                    episode_length=_as_int(_value_at_index(episode_lengths, traj_index)),
                    success=self._success_bool(success_metrics),
                    tool_call_count=_as_int(_value_at_index(tool_callings, traj_index)),
                    steps=steps,
                    success_metrics=success_metrics,
                    metadata={
                        "schema_version": EXPORT_SCHEMA_VERSION,
                        "source": "verl-agent TrajectoryCollector.vanilla_multi_turn_loop",
                        "done_is_inferred_from_episode_length": True,
                    },
                )
            )

        return records

    def export(
        self,
        total_batch_list: list[list[dict[str, Any]]],
        total_infos: list[list[dict[str, Any]]],
        episode_rewards: Any,
        episode_lengths: Any,
        success: dict[str, Any],
        traj_uid: Any,
        tool_callings: Any,
        overwrite: bool = False,
    ) -> int:
        records = self.build_records(
            total_batch_list=total_batch_list,
            total_infos=total_infos,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            success=success,
            traj_uid=traj_uid,
            tool_callings=tool_callings,
        )
        return JsonlWriter(self.export_config.export_path, overwrite=overwrite).write_many(records)

    @staticmethod
    def _success_metrics_for_index(success: dict[str, Any], index: int) -> dict[str, Any]:
        if not isinstance(success, dict):
            return {}
        return {str(key): _value_at_index(value, index) for key, value in success.items()}

    @staticmethod
    def _success_bool(success_metrics: dict[str, Any]) -> bool | None:
        if not success_metrics:
            return None
        preferred_keys = ["success_rate", "success", "won"]
        for key in preferred_keys:
            if key in success_metrics:
                return _as_bool(success_metrics[key])
        for key, value in success_metrics.items():
            if "success" in key or "won" in key:
                return _as_bool(value)
        return None

