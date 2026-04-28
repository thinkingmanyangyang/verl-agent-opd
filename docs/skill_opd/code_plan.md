# 代码实现路线

更新时间：2026-04-28

## 当前主体

当前以 `D:\codex_workspace\skill_opd\verl-agent` 作为主开发仓库。

`resource/` 只保留参考资料和参考仓库。旧的 `external/`、`src/skill_opd`、`scripts/`、`tests/`、`patches/` 路线已经删除。

## 第一阶段目标

第一阶段只实现离线 student rollout 导出：

```text
verl-agent 原生 rollout
-> total_batch_list + total_infos
-> Skill-OPD exporter
-> raw_trajectories.jsonl
```

暂时不做：

- skill retrieval
- teacher scoring
- residual gate
- distill loss
- trainer loss 修改
- student prompt skill injection

## 第一阶段代码结构

```text
verl-agent\
  agent_system\
    skill_opd\
      __init__.py
      README.md
      config.py
      schema.py
      io.py
      rollout_exporter.py
      rollout_hook.py
  examples\
    skill_opd\
      README.md
      prepare_sokoban_data.sh
      download_qwen3_model.sh
      run_sokoban_rollout_export_qwen3.sh
```

只修改一个 verl-agent 原文件：

```text
verl-agent\agent_system\multi_turn_rollout\rollout_loop.py
```

修改方式是在 `vanilla_multi_turn_loop()` 完成 `success = envs.success_evaluator(...)` 后、`return` 前调用：

```python
maybe_export_rollout(
    config=self.config,
    tokenizer=self.tokenizer,
    total_batch_list=total_batch_list,
    total_infos=total_infos,
    episode_rewards=episode_rewards,
    episode_lengths=episode_lengths,
    success=success,
    traj_uid=traj_uid,
    tool_callings=tool_callings,
)
```

这比 step-level hook 更合理，因为 verl-agent 已经在内部维护完整轨迹。

## 第一阶段导出字段

trajectory 级别：

```text
trajectory_id
env_name
total_reward
episode_length
success
tool_call_count
steps
metadata
```

step 级别：

```text
step_id
prompt_text / h_t
response_text
response_token_ids
reward
active_mask
done
info
is_action_valid
```

字段来源：

- `total_batch_list`：prompt、response token、reward、active mask、traj uid 等。
- `total_infos`：env info、done/success 相关环境信息。
- `episode_rewards / episode_lengths / success / tool_callings`：trajectory summary。

## 第一阶段运行脚本

当前不新增独立 rollout engine，而是复用 verl-agent 的 `verl.trainer.main_ppo` 入口。原因是模型加载、vLLM rollout worker、Sokoban env manager、prompt 构造都已经在原框架里打通。

```bash
cd D:\codex_workspace\skill_opd\verl-agent
bash examples/skill_opd/prepare_sokoban_data.sh
bash examples/skill_opd/download_qwen3_model.sh
bash examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

第三个脚本默认 `trainer.val_only=True`，只导出 validation rollout；当前机器没有 GPU，所以本轮只验证语法和 exporter fake 数据。

## 后续阶段

第二阶段：

```text
raw_trajectories.jsonl -> skill_memories.json
```

参考 SkillRL 的 `memory_data/generated_memories_*.json`，但需要保留 raw token/mask 信息，不能只保留抽象 memory。

第三阶段：

```text
skill_memories.json -> global_skills.json / residual_skills.json
```

第四阶段：

```text
teacher scoring:
p^g, p^{g+r}, m_t, alpha_t, z*
```

第五阶段：

```text
distill loss -> verl trainer
```

## 参考代码

- rollout 主链路：`verl-agent/agent_system/multi_turn_rollout/rollout_loop.py`
- env/prompt/projection：`verl-agent/agent_system/environments/*`
- SkillRL memory schema：`resource/repos/SkillRL/memory_data/*/generated_memories_*.json`
- SkillRL skill generation：`resource/repos/SkillRL/skill_generation/*.py`
