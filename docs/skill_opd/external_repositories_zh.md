# 参考资源管理

更新时间：2026-04-28

当前项目不再采用 `external/verl_agent` 作为外部依赖式集成。

## 1. 主体仓库

主开发仓库已经移动到：

```text
D:\codex_workspace\skill_opd\verl-agent
```

该目录是后续开发、运行和实验的主体。

## 2. 参考资源

参考论文、网页和其他仓库统一放在：

```text
D:\codex_workspace\skill_opd\resource
```

`resource/` 只用于阅读和对比，不参与运行主体，也不会提交。

当前重要参考：

```text
resource\repos\SkillRL
```

SkillRL 的作用：

- 参考 `memory_data/generated_memories_*.json` 的 memory schema。
- 参考 `skill_generation/*.py` 的 skill generation prompt。
- 参考 `SkillsOnlyMemory` 的 skill bank 格式和检索逻辑。

不采用的部分：

- 不采用 SkillRL 的 student prompt skill injection 作为主方法。
- 不把 SkillRL 当训练主框架。
- 不依赖 SkillRL 生成 raw rollout，因为它没有公开完整 raw trajectory 到 memory data 的生成闭环。

## 3. 已废弃方案

以下方案已经废弃：

```text
external\verl_agent\
src\skill_opd\integrations\verl_agent\
patches\
scripts\setup\apply_verl_agent_sokoban_export_patch.*
```

原因：

- 这会把项目变成“外部 adapter 包”，不符合现在以 `verl-agent` 为主体开发的方向。
- 后续接 response mask、logprob、DataProto 和 trainer hook 会变得绕。
- rollout 数据本来已经由 verl-agent 的 `TrajectoryCollector` 收集，不应在外部重复实现。
