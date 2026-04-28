# Baseline 与 Ablation 计划

更新时间：2026-04-24

## 1. 评测目标

这套 baseline/ablation 不是为了“凑表”，而是要逐层回答下面四个关键问题：

1. `skill` 本身是否比无 skill 更有帮助？
2. 仅有 global skill 是否已经足够？
3. step skill 若存在，是否必须是 `state-dependent residual`，而不是 always-on second skill？
4. 你的收益到底来自 `retrieval`、`gate`、还是 `residual target construction`？

## 2. 主 baseline

### 2.1 必做 baseline

| Baseline | 定义 | 主要回答的问题 | 备注 |
| --- | --- | --- | --- |
| No Skill | plain GRPO / plain RL post-training，无 teacher privileged context | 不使用 skill 时的基础性能是多少 | 是最低对照，不是蒸馏方法 |
| Vanilla OPD | student rollout，teacher 只看 plain prompt，不看额外 context | 单纯 on-policy distillation 本身带来多少收益 | 对照 “有无 privileged teacher context” |
| OPSD | teacher 看 ground-truth / verified solution 等 privileged trace，student 只看问题 | privileged teacher 是否本身有效 | 只适用于 reasoning / small env，不适合 AppWorld 主实验 |
| OPCD | teacher 看 generic context / experience，student 只看 plain prompt | generic context-conditioned teacher 能否提升 | 是你方法最重要的范式对照 |
| Skill-SD | teacher 看 trajectory-derived global skill；student plain prompt | global trajectory skill 是否足够 | 你方法的最直接父 baseline |
| OEL | 经验抽取 + consolidation 循环 | 在线 experience 提取/整合是否更强 | 更适合 text games；AppWorld 可选，不是首轮必做 |

### 2.2 baseline 的推荐落地优先级

按优先级排序：

1. `No Skill`
2. `Vanilla OPD`
3. `OPCD`
4. `Skill-SD`
5. `Hierarchical Residual Skill OPD`
6. `OEL`（若先做 FrozenLake / Sokoban）
7. `OPSD`（仅在有 ground-truth solution 的 reasoning sanity-check 上做）

原因：

- `No Skill` / `Vanilla OPD` 给出最基础增益拆分。
- `OPCD` 检验 “generic context 已经够不够”。
- `Skill-SD` 检验 “global skill 是否已经够不够”。
- 你的方法需要证明：不是 generic context 造成收益，而是 `hierarchical + residual + sparse gate`。

## 3. 关键 ablation

### 3.1 结构性 ablation

| Ablation | 定义 | 回答什么问题 |
| --- | --- | --- |
| Global Skill Only | teacher 只看 `g`，无 `r_t` | global skill 单独是否已足够 |
| Step Skill Only | teacher 只看 `r_t`，无 `g` | local step skill 能否替代长期先验 |
| Global + Always-on Step Skill | 每个 step 都加 `r_t` | gate 是否真的必要；prompt 膨胀是否会伤害稳定性 |
| Global + Gated Residual Step Skill | 你的完整方法 | 稀疏 residual patch 是否优于 always-on |

### 3.2 retrieval ablation

| Ablation | 定义 | 回答什么问题 |
| --- | --- | --- |
| Query-only Retrieval | `q -> r_t`，不看状态 | residual 是否必须状态依赖 |
| Query + Step-state Retrieval | `(q, h_t) -> r_t` | step-state 是否能提供足够局部信息 |
| Query + Step-state + Global Retrieval | `(q, h_t, g) -> r_t` | residual 是否应该显式建模为“在 global skill 前提下还缺什么” |
| Direct Single Residual Retrieval | 不做 top-k candidates summarize，直接取一条 residual | candidate retrieval + summarization 是否真的更稳 |
| Candidate Retrieval + Residual Summarization | 你的推荐方案 | 多候选再总结是否能减少噪声与重复 skill |

### 3.3 target construction ablation

| Ablation | 定义 | 回答什么问题 |
| --- | --- | --- |
| Direct Teacher+Skill Distill | 直接把 `p_t^{g+r}` 当 target | residual patch 是否必须以 global teacher 为基底 |
| Residual Target Distill | 用 `z_t^* = z_t^g + \alpha_t^*(z_t^{g+r}-z_t^g)` | 差分 target 是否比直接复制 residual teacher 更稳 |
| Hard Gate | `\alpha_t^* ∈ {0,1}`，只做 top-M | soft blending 是否必要 |
| Soft Stop-grad Gate | 连续 `\alpha_t^*`，但 stop-gradient | 软残差是否比硬开关更平滑 |
| Learnable Gate | 直接学习 `\alpha_t` | 验证是否会出现 collapse，作为反证 ablation |

### 3.4 benefit/gate ablation

| Ablation | 定义 | 回答什么问题 |
| --- | --- | --- |
| Entropy-only Gate | 只看 `H(p_t^g)` | 证明固定 entropy threshold 不稳 |
| Entropy-drop Only | 只看 `[H(p_t^g)-H(p_t^{g+r})]_+` | JS 项是否必要 |
| JS-only | 只看 `JS(p_t^{g+r} || p_t^g)` | 分布变化但不降熵是否足够 |
| Full Benefit Score | `entropy-drop + rho * JS` | 两部分结合是否最稳 |
| Per-trajectory Top-M | 每条轨迹固定预算 | budget control 是否足够 |
| Global Threshold | 全局固定阈值 | 跨任务/跨prompt 是否明显不稳 |

## 4. 环境分阶段计划

### 4.1 当前阶段：CPU / 离线原型

只做 teacher-side 统计，不做完整训练：

- FrozenLake / Sokoban trajectory dump
- 小规模 reasoning trajectory dump
- skill retrieval / residual scoring / gate 分析

主要指标：

- candidate step 比例
- activated residual step 比例
- `m_t` 分布
- `alpha_t^*` 分布
- `KL(student || global target)` 与 `KL(student || residual target)` 的差值
- 按 step type 的 benefit 统计

### 4.2 有 GPU 后：小环境训练优先

第一阶段建议先上：

1. FrozenLake
2. Sokoban
3. 小规模 reasoning（若有 verified solution）

原因：

- horizon 可控；
- step type 可解释；
- retrieval / gate 机制更容易 debug；
- 能快速区分 “residual 真有用” 与 “只是 prompt 噪声”。

### 4.3 第二阶段：AppWorld

AppWorld 值得作为后续主 benchmark，但不应是第一枪：

- 它更适合验证 long-horizon agent generalization；
- 不适合一开始排查 skill decomposition、gate stability、retrieval 设计。

## 5. 我建议最终主表怎么讲故事

### 5.1 主表

主表建议只保留：

- No Skill
- Vanilla OPD
- OPCD
- Skill-SD
- Hierarchical Residual Skill OPD

这样叙事最清楚：

- 无 teacher privileged context
- generic context
- global skill
- hierarchical residual skill

### 5.2 Ablation 表

单独做三张 ablation 表：

1. 结构 ablation：global / step / always-on / gated residual
2. retrieval ablation：query-only / q+h / q+h+g / summarize
3. target/gate ablation：direct distill / residual target / hard gate / soft gate / learnable gate

### 5.3 关键反证实验

必须有两个反证：

1. `Global Skill Only` 很强，但在关键 step 上仍有可测 teacher-side residual benefit。
2. `Learnable α_t` 容易塌缩到接近 0，而 detached teacher-side gate 更稳定。

这两个实验直接对应你的核心理论点。

## 6. 需要特别注意的公平性

做对比时必须保证：

- student rollout owner 一致；
- teacher model size 一致；
- teacher/student chat template 一致或清晰记录差异；
- retrieval budget 一致；
- residual steps 预算 `M` 明确控制；
- 不能让你的方法偷偷多用很多 teacher compute。

建议所有方法统一报告：

- 平均每轨迹 teacher pass 数；
- 平均额外 context token 数；
- 平均被激活 residual step 数。

## 7. 我对 baseline 设计的结论

如果主问题是：

> step skill 是否真的必要？

那么真正必须比较的不是 “有没有 step skill”，而是：

- `global only`
- `global + always-on step skill`
- `global + gated residual step skill`

因为只有这三者才能回答：

1. global skill 是否已经足够；
2. local skill 是否确实有边际价值；
3. 这个边际价值是否必须通过 sparse residual 才能稳定释放出来。

## Sources

- Skill-SD: [arXiv:2604.10674](https://arxiv.org/abs/2604.10674)
- OPCD: [arXiv:2602.12275](https://arxiv.org/abs/2602.12275), [LMOps/opcd](https://github.com/microsoft/LMOps/tree/main/opcd)
- OEL: [arXiv:2603.16856](https://arxiv.org/abs/2603.16856), [LMOps/oel](https://github.com/microsoft/LMOps/tree/main/oel)
- OPSD: [arXiv:2601.18734](https://arxiv.org/abs/2601.18734), [repo](https://github.com/siyan-zhao/OPSD)
- G-OPD: [arXiv:2602.12125](https://arxiv.org/abs/2602.12125), [repo](https://github.com/RUCBM/G-OPD)
- OPD recipe analysis: [thunlp/OPD](https://github.com/thunlp/OPD)
