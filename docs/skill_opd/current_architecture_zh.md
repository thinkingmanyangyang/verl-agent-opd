# 当前代码架构

更新时间：2026-04-28

## 1. 当前结论

本项目现在采用 **verl-agent 作为主体框架** 的方案。

也就是说，后续真正运行、修改和实验的主仓库是：

```text
D:\codex_workspace\skill_opd\verl-agent
```

`resource/` 只作为论文、网页、参考仓库缓存，不作为运行主体。

## 2. 顶层目录

```text
D:\codex_workspace\skill_opd\
  verl-agent\          # 主开发仓库，基于 langfengq/verl-agent
  resource\            # 参考资料和参考代码，只读
  docs\                # 当前项目设计文档
  research_note.md
  method_design.md
  baseline_plan.md
  code_plan.md
  todo_next.md
```

已经废弃并删除的旧路线：

```text
external\verl_agent\   # 旧外部仓库路径
src\skill_opd\         # 旧 adapter 路线
scripts\               # 旧 patch / mock 脚本
tests\                 # 旧 adapter 单测
patches\               # 旧 patch 方案
```

## 3. 开发原则

当前阶段不做完整训练，不改 teacher，不改 loss。

第一阶段只做：

```text
verl-agent 原生 student rollout
-> total_batch_list + total_infos
-> rollout 完成后导出
-> raw_trajectories.jsonl
```

核心原则：

- 不改 environment 语义。
- 不往 student prompt 注入 skill。
- 不在每个 step 里截流一大堆参数。
- 不改 PPO / GRPO / GiGPO loss。
- 优先复用 verl-agent 已经收集好的 `total_batch_list` 和 `total_infos`。
- Skill-OPD 代码放在 `agent_system/skill_opd/`，不要散进大文件。

## 4. 第一阶段新增结构

第一阶段只需要在 `verl-agent` 内新增一个轻量目录：

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

文件职责：

- `config.py`：从 OmegaConf / dict 中读取 `skill_opd` 配置。
- `schema.py`：定义 `TrajectoryRecord` 和 `StepRecord`。
- `io.py`：JSONL 写入和 dataclass 序列化。
- `rollout_exporter.py`：把 `total_batch_list + total_infos` 转成 trajectory JSONL。
- `rollout_hook.py`：给 `rollout_loop.py` 调用的薄 hook；未开启时直接 return。
- `examples/skill_opd/prepare_sokoban_data.sh`：复用 `examples.data_preprocess.prepare` 准备 text-mode Sokoban 占位数据。
- `examples/skill_opd/download_qwen3_model.sh`：下载 `Qwen/Qwen3-4B` 或 `Qwen/Qwen3-8B`。
- `examples/skill_opd/run_sokoban_rollout_export_qwen3.sh`：用 verl-agent 原 PPO 入口做 validation-only rollout 导出。

暂时不提前拆出 `teacher/`、`distill/`、`skills/` 子目录。等代码量变大再拆。

## 5. 唯一主干改动点

第一阶段只轻微修改：

```text
verl-agent\agent_system\multi_turn_rollout\rollout_loop.py
```

调用位置在 `TrajectoryCollector.vanilla_multi_turn_loop()` 末尾：

```python
success = envs.success_evaluator(...)

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

注意：不采用 `maybe_export_rollout_step(...)`。原因是 verl-agent 已经在 loop 内完成了轨迹收集，应该在完整 rollout 结束后导出。

## 6. 配置

第一阶段配置保持最小：

```yaml
skill_opd:
  export_rollouts: false
  export_path: outputs/skill_opd/rollouts/sokoban.jsonl
  include_token_ids: true
  include_text: true
  include_infos: true
  overwrite: false
```

默认关闭，关闭时不影响 verl-agent 原训练和验证流程。

当前导出脚本使用：

```bash
bash examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

该脚本设置 `trainer.val_only=True`，所以用途是生成离线验证 rollout，不做参数更新。但它仍依赖 GPU / vLLM / Ray，本机 CPU 环境只做静态检查。

## 7. 后续阶段

后续按顺序推进：

1. `raw_trajectories.jsonl -> skill_memories.json`
2. `skill_memories.json -> global_skills.json / residual_skills.json`
3. 离线 teacher scoring：`p^g`、`p^{g+r}`、`m_t`、`alpha_t`、`z*`
4. distill loss 接入 `verl` trainer

SkillRL 只作为 skill memory / skill generation 的参考，不作为运行主体。
