# Hierarchical Residual Skill OPD 研究笔记

更新时间：2026-04-24  
本笔记基于当天核实过的论文、项目页与公开仓库整理，不假设任何此前对话上下文。

## 1. 问题定义

目标不是再做一个“skill prompt engineering”技巧，而是提出一个更严格的、以 `skill` 为中心的后训练方法：

- `global skill` 提供任务级长期策略先验。
- `step skill` 不预设总是必要。
- 只有当当前状态下 `global skill` 不足以解释 teacher 决策时，才引入 `residual step skill`。
- `residual step skill` 不是第二份完整 skill，而是对 `global skill` 的局部修正。
- teacher 只在少量关键 step 上接收 residual patch，再通过 on-policy distillation 蒸馏回 student。

这条叙事必须正面回应专家质疑：

> step-level skill 不预设必要；它只在当前状态下 global skill 不足以解释 teacher 决策时才出现，因此应建模为 residual correction，而不是独立第二份完整 skill。

## 2. 相关工作脉络

### 2.1 总览关系

| 工作 | teacher 额外信息 | rollout owner | 训练信号 | 与本方法关系 | 公开代码状态 |
| --- | --- | --- | --- | --- | --- |
| Skill-SD | 轨迹总结出的 global natural-language skill，仅 teacher 可见 | student | importance-weighted reverse-KL | 最直接父工作；提供 teacher-only skill conditioning 范式 | 找到论文与项目页，2026-04-24 未发现清晰的官方训练 repo |
| OPSD | ground-truth / verified reasoning trace 等 privileged context | student | token-level divergence | 说明 privileged teacher + student-owned rollout 可行 | `siyan-zhao/OPSD`，基于 TRL GOLD trainer |
| OPCD | generic context / experience / system prompt | student | reverse-KL on student rollouts | 你的方法本质上仍是 context-conditioned teacher，只是 context 被结构化为 hierarchical skill | `microsoft/LMOps/opcd` |
| OEL | user-side deployment 轨迹中提取的 experience | deployment user side -> consolidate to student | extract + consolidate loop | 最接近 “experience -> retrieval -> summary -> context distill” 的工程路线 | `microsoft/LMOps/oel` |
| G-OPD / ExOPD | flexible reference + reward scaling | student | dense KL-constrained RL 视角 | 给出“teacher 相对 baseline/reference 的改进方向”这一理论视角 | `RUCBM/G-OPD` |
| OPD / GKD | teacher 对 student 自采样轨迹重打分 | student | token-level KD | 提供 on-policy distillation 的基础范式 | 经典论文为 GKD；近期工程实现见 `thunlp/OPD` |
| VeRL / HybridFlow | 无 | 无 | 基础设施 | 最现实的训练底座候选 | `verl-project/verl` |
| AppWorld | 无 | 无 | benchmark / environment | 后续主 benchmark 候选，但不适合本轮 CPU-only 原型 | `StonyBrookNLP/appworld` |

### 2.2 Skill-SD 的关键贡献与不足

`Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents` 的核心价值有四点：

1. 证明了 trajectory-derived skill 作为 teacher-side privileged information 是有效的。
2. 保持 student-owned rollout，避免离线 teacher 轨迹分布错配。
3. 用 importance-weighted reverse-KL 稳定蒸馏。
4. 通过动态 teacher 同步降低 teacher/student 漂移。

但它仍主要是：

- 一条压缩 natural-language global skill；
- 一个 teacher-only skill-conditioned distillation 机制。

它没有显式回答：

- global skill 与 step-specific skill 的层次关系是什么；
- step-level skill 是否真的必要；
- 若必要，如何让它成为状态依赖的 residual correction；
- 如何避免每个 step 都把 skill 堆进 prompt。

### 2.3 OPCD 与 OEL 对本方法的直接启发

`OPCD` 的统一视角很重要：teacher 可以多看 context，student 只在 plain prompt 下 rollout，然后用 reverse-KL 去蒸馏 context-conditioned teacher。  
你的方法仍属于这个范式，但 context 不再是 generic context，而是：

- `global skill g`
- `residual step skill r_t`
- 并且 `r_t` 是相对 `g` 的 patch

`OEL` 给出的更关键启发是工程流程：

- 先从交互轨迹中抽取可迁移 experience；
- 再做 consolidation；
- 并且经验提取优于直接喂原始轨迹。

这对你的 residual retrieval 很重要，因为它支持如下设计：

- 不直接从 `(q, h_t)` 生硬检索一条 step skill；
- 而是先检索多个候选 residual memories；
- 再在 `(q, h_t, g, candidates)` 条件下总结出当前 step 的专用 residual patch。

### 2.4 G-OPD 对你的启发

`G-OPD` 最值得借的不是具体 reward scaling，而是这条思想：

- teacher 分布变化本身可以视为 dense training signal；
- 真正重要的不是“复制 teacher”，而是“teacher 相对一个 baseline/reference 的改进方向”。

你的方法正好可把这个思想改写为：

- baseline/reference 不是 teacher base model；
- 而是 `global-only teacher`
- residual 的价值来自 `global+residual teacher` 相对 `global-only teacher` 的增量。

这使你的 residual 目标天然可以写成差分形式，而不是直接复制 `p_t^{g+r}`。

### 2.5 VeRL 与 AppWorld 的现实地位

核实后的 `verl-project/verl` 是当前活跃主仓库，`2026-04-23` 仍在更新，文档中已经明确支持：

- PPO/GRPO 等 post-training；
- `examples/sglang_multiturn` 多轮 tool-calling；
- `verl/experimental/agent_loop`；
- `multi-turn tokenization`、`tool config`、`MCP` 等相关文档。

但它同时也明确表明：

- 多轮 agent loop 仍有实验性组件；
- 真正的环境交互与 agent integration 仍在持续演进。

所以它适合作为训练底座，但你自己的 method 逻辑不应该一开始就硬嵌进一个大而全的 agent framework。

`AppWorld` 则非常适合作为后续主 benchmark：

- 9 类 app；
- 457 APIs；
- 750 interactive coding tasks；
- 可程序化评测，且能检查 collateral damage。

但它不适合本轮原型，因为：

- 环境重；
- horizon 长；
- benchmark 工程成本明显高于先在小环境上验证 residual-skill 机制。

## 3. 你的方法真正要解决什么

你要解决的不是 “如何再多给 teacher 一段提示词”，而是以下更严格的问题：

1. 如何把 `skill` 从单条自然语言总结，提升为层次化、功能化的训练对象。
2. 如何把长期策略先验与局部状态修正明确分开。
3. 如何在不让 prompt 爆炸的前提下，把局部 skill 只用在必要 step。
4. 如何把这种局部修正学回 student，而不是永远留在 teacher prompt 里。

对应地，方法核心应被表述为：

- `global skill` 负责“通常应该怎样做”。
- `residual step skill` 负责“在这个状态下，和通常做法相比，此刻需要补哪一小块修正”。

## 4. 为什么 step skill 必须是 residual patch

这是整个方法最关键的理论立足点。

### 4.1 不预设必要性

如果某个 step 上 `global skill` 已经足够解释 teacher 的选择，那么合理结果应该是：

- `p_t^{g+r} ≈ p_t^g`
- `m_t ≈ 0`
- `α_t^* ≈ 0`

也就是说，好的方法不应该强迫每个 step 都“需要 local skill”，而应该允许 residual 自动失活。

### 4.2 避免 duplicated skill

如果把 step skill 当成第二份完整 skill，会出现三个问题：

1. 内容重复：step skill 重新复述全局策略。
2. 作用边界模糊：teacher 到底是在听 global 还是在听 local。
3. prompt 累积：每个 step 都在附加一段几乎完整的新 skill。

把 step skill 改成 residual patch 后，局部 skill 的语义就变成：

> 在已经有 global skill 的前提下，这个状态还缺的那一点额外修正。

### 4.3 更符合 teacher 差分建模

你的目标式：

\[
z_t^* = z_t^g + \alpha_t^* ( z_t^{g+r} - z_t^g )
\]

本质上就是把 residual 当作 logits 上的局部增量。  
这和 “teacher 相对 baseline 的改进方向” 完全一致。

## 5. 为什么 fixed entropy threshold 不稳

“高熵就开 residual、低熵就不开 residual” 是不稳的，原因至少有五个：

1. entropy 不同任务族不校准：工具调用、自由生成、验证、修复步骤的天然 entropy 基线不同。
2. entropy 受 prompt 模板影响：chat template、thinking template、system prompt 都会改变分布温度。
3. entropy 受 tokenization 影响：多轮 agent 输出里，风格 token 与工具 token 的熵结构差别很大。
4. entropy 随训练漂移：teacher 同步后，同一阈值的语义会变。
5. entropy 高不代表 residual 有用：teacher 可能不确定，但 residual 也未必让分布更好。

因此更合理的做法是两层机制：

- 第一层只做候选筛选：高熵 step 或关键 step type 进入候选集。
- 第二层才做 residual benefit 判定：比较 `p_t^g` 与 `p_t^{g+r}` 的变化是否值得。

## 6. 为什么自由可学的 α_t 容易塌缩

如果 `α_t` 是直接可学参数，而 teacher logits 又是 detached 的，那么优化会天然偏向“让当前 loss 更容易下降”，而不是“让 residual 真正有用”。

典型塌缩机制是：

1. `z_t^g` 已经给出一个稳定 teacher target。
2. 引入 residual 后，teacher target 会偏移，student 需要额外追这个偏移。
3. 如果再叠加 sparsity penalty，最简单的解就是把 `α_t -> 0`。

更重要的是：在没有 verifier / reward / counterfactual label 的纯 OPD 场景里，
`KL + sparse loss` 监督到的其实是“这个 residual 是否好学”，而不是“这个 residual 是否真有价值”。

所以自由可学 `α_t` 会错误地把 gate 学成：

- 对 student 越难跟随的 residual，越倾向于关掉；
- 而不是对 global teacher 改进越大的 residual，越倾向于打开。

## 7. 为什么 detached teacher-side gate 更稳

更稳的做法是先在 teacher 侧计算 residual marginal benefit：

\[
m_t = [ H(p_t^g) - H(p_t^{g+r}) ]_+ + \rho \cdot JS(p_t^{g+r} \| p_t^g)
\]

再把它映射成 stop-gradient 的 gate target：

\[
\alpha_t^* = \text{stopgrad}\left(\sigma(\kappa(\text{normalize}(m_t)-\tau_m))\right)
\]

这样有三个好处：

1. `是否启用 residual` 由 teacher-side 增益决定，而不是 student 当前是否容易拟合。
2. gate 不会被 KL 和 sparsity loss 直接拉向 0。
3. 可以自然叠加 budget 控制，例如每条轨迹只保留 top-`M` 个 residual step。

如果以后一定要做可学习 gate，更合理的方式也不是直接让它吃 KL，而是：

- 先用 teacher-side `m_t` 或其 top-`M` 标记生成监督标签；
- 再单独训练一个 gate predictor 去预测这些标签；
- 训练时仍把 predictor 输出 stop-gradient 后用于是否运行 residual teacher。

## 8. 本方法的主要风险

### 8.1 无法直接声称“识别了真正的 skill usefulness”

在没有外部 reward、verifier 或 counterfactual rollout 时，你能稳妥测到的是：

- teacher-side shift；
- teacher-side marginal benefit；
- influence。

不能在第一版里把它说成“真实有用性”。

### 8.2 retrieval / summarization 可能泄漏成第二份完整 skill

如果 residual summarizer 没有被严格约束，它可能：

- 重复 global skill；
- 输出完整重规划；
- 甚至引入与 global skill 冲突的信息。

所以 residual skill 需要有明确模板和长度预算。

### 8.3 teacher 成本增加

第一版必须坚持：

- 1 次 global teacher pass；
- `M` 次 residual teacher rescore；
- 且 `M << T`。

不能直接做全步 residual scoring。

### 8.4 小环境有效不等于 AppWorld 立即有效

FrozenLake / Sokoban 更适合先验证：

- residual 是否稀疏；
- gate 是否稳定；
- retrieval 是否比 query-only 更合理。

但 AppWorld 的 step type、tool semantics、代码执行结构都更复杂，迁移仍需单独验证。

## 9. 本方法的预期优势

1. 理论叙事更强：正面回应 “step skill 是否必要” 这个核心质疑。
2. 结构更清楚：global 负责长期先验，residual 负责局部修正。
3. prompt 更可控：不累积历史 step skills，只用固定槽位。
4. 计算更可控：通过 candidate screening + top-`M` gating 把成本限制在 `1+M`。
5. 更利于分析：可以直接统计 residual 主要出现在哪些 step type、哪些位置、带来多大 teacher-side shift。
6. 更利于未来工程优化：teacher 输入结构固定后，更容易做 KV cache 复用和局部重打分。

## 10. 我对当前方向的结论

这个方向是成立的，但前提是叙事必须非常克制：

- 不是“每个 step 都需要 local skill”；
- 不是“我们评估了 skill 真 usefulness”；
- 不是“再喂更多 prompt 就更好”。

更准确的说法应该是：

> 我们把 trajectory-level skill 从单层 global summary，推进为“global prior + sparse state-dependent residual patch”的层次化 teacher context；只在少量 teacher-side benefit 明确的关键 step 上启用 residual，并通过 on-policy distillation 把这种局部修正学回 student。

## Sources

- Skill-SD paper: [arXiv:2604.10674](https://arxiv.org/abs/2604.10674), [project page](https://k1xe.github.io/skill-sd/)
- OPSD paper: [arXiv:2601.18734](https://arxiv.org/abs/2601.18734), [repo](https://github.com/siyan-zhao/OPSD)
- OPCD paper: [arXiv:2602.12275](https://arxiv.org/abs/2602.12275), [repo](https://github.com/microsoft/LMOps/tree/main/opcd)
- OEL paper: [arXiv:2603.16856](https://arxiv.org/abs/2603.16856), [repo](https://github.com/microsoft/LMOps/tree/main/oel)
- G-OPD / ExOPD paper: [arXiv:2602.12125](https://arxiv.org/abs/2602.12125), [repo](https://github.com/RUCBM/G-OPD)
- GKD / early OPD paper: [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)
- Rethinking OPD recipe repo: [thunlp/OPD](https://github.com/thunlp/OPD)
- VeRL / HybridFlow: [arXiv:2409.19256](https://arxiv.org/abs/2409.19256), [repo](https://github.com/verl-project/verl), [docs](https://verl.readthedocs.io/en/latest/)
- AppWorld paper: [arXiv:2407.18901](https://arxiv.org/abs/2407.18901), [repo](https://github.com/StonyBrookNLP/appworld), [website](https://appworld.dev/)
