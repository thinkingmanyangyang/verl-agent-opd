# 下一步 To-do

更新时间：2026-04-28

## 当前优先级

现在先完成第一阶段：**基于 verl-agent 原生 rollout 的离线 trajectory 导出**。

## 第一阶段已完成

1. 在 `verl-agent/agent_system/skill_opd/` 新建最小导出模块。
2. 实现 trajectory / step schema、JSONL writer、config reader、rollout exporter、rollout hook。
3. 在 `TrajectoryCollector.vanilla_multi_turn_loop()` 完成 `success_evaluator` 后接入导出 hook。
4. 新增 Sokoban 数据准备、Qwen3 模型下载、validation-only rollout 导出脚本。
5. 完成 `py_compile`、bash 语法检查、fake rollout JSONL 导出测试。

## 下一批 10 件事

1. 在 GPU 机器上用 `Qwen/Qwen3-4B` 跑 1 个极小 validation-only rollout。
2. 检查真实 JSONL 字段是否包含后续 teacher scoring 需要的 prompt / response token。
3. 如果真实输出缺少精确 `done`，再最小修改 rollout loop 额外记录 `dones`。
4. 写 `raw_trajectories.jsonl -> skill_memories.json` 的转换脚本。
5. 参考 SkillRL 的 generated memories 格式，设计 Sokoban global skill memory schema。
6. 实现最小 global skill summarizer，占位可用人工模板或 API。
7. 实现 residual candidate step selector，先用 entropy/invalid-action/error-info 做候选。
8. 设计离线 teacher scoring 输入格式，不进入训练。
9. 在 fake logits 上验证 `m_t` 和 `alpha_t` 计算。
10. 再决定是否把 exact response mask/exported log prob 加入 schema。

## 最大 blocker

当前 blocker 不是 teacher scoring，而是先把 student rollout 的原始数据导出来，并保证字段足够后续使用：

```text
prompt / h_t
response text
response token ids
reward
active mask
info
trajectory id
episode summary
```

如果第一阶段只导出抽象 memory，会丢失 OPD 后续需要的 token 对齐信息；如果只导出 DataProto，又不方便 skill generation。所以第一阶段先导出 raw trajectory JSONL。

## 下一阶段

第一阶段完成后再做：

1. `raw_trajectories.jsonl -> skill_memories.json`
2. 参考 SkillRL 生成 `global_skills.json`
3. 设计 Sokoban residual skill schema
4. 离线 teacher scoring
5. distill loss 接入 trainer
