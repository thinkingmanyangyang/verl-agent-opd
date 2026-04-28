# Hierarchical Residual Skill OPD 方法设计

更新时间：2026-04-27

## 1. 方法一句话

对每个任务先给 teacher 一个 `global skill` 作为长期策略先验；  
再只在少量关键 step 上，引入相对 `global skill` 的 `residual step skill`，并把 `global-only teacher` 与 `global+residual teacher` 的差值蒸馏回 student。

## 2. 整体五步流程

### Step 1：收集 student on-policy trajectories

student 在环境中自主 rollout，不给任何 skill。

```text
q -> student rollout -> τ = {(h_t, a_t, o_t, reward_t)}
```

每一步必须记录：

- `q`：任务 query
- `h_t`：step 前历史 / 状态 / observation context
- `a_t`：student action / response
- `o_t`：environment feedback
- `action_token_range`：哪些 token 是 student action
- `response_mask`：哪些 token 参与训练 loss
- `reward / done / success`

这一阶段的目标是得到 step-aligned trajectory schema，而不是训练。

### Step 2：从历史轨迹构建 skill bank

从历史成功或高质量 trajectories 中离线构建两类 skill：

```text
global skill g:
任务级长期策略先验。

residual skill candidate:
局部状态补丁，只描述 global skill 已存在时当前 step 还缺什么。
```

关键约束：

- `global skill` 负责长期策略和任务级结构。
- `residual skill` 不是第二份完整 skill。
- `residual skill` 必须能被表述为局部 patch，例如 local constraint、tool error、verification、replan、state update。

### Step 3：对 student trajectory 做 teacher scoring

对每条 trajectory 先检索一个 global skill：

```text
q -> g
```

然后对所有 step 计算 global-only teacher：

```text
p_t^g = π_T(. | q, g, h_t)
```

只在少量候选 step 上做 residual scoring。候选 step 来自：

- high-rank uncertainty / entropy step
- tool call
- replan
- verification
- error handling
- invalid action / environment warning

对候选 step：

```text
(q, h_t, g) -> top-k residual candidates
summarize(q, h_t, g, candidates) -> r_t
p_t^{g+r} = π_T(. | q, g, r_t, h_t)
```

第一版必须保持 `1 + M` 次 teacher scoring，其中 `M << T`。

### Step 4：计算 residual benefit 与 gate

不默认 step skill 有用，而是用 teacher-side residual benefit 判断：

```text
m_t = [H(p_t^g) - H(p_t^{g+r})]_+ + ρ JS(p_t^{g+r} || p_t^g)
```

含义：

- residual 让 teacher 更确定。
- residual 真的改变了 teacher 分布。
- 两者同时出现时，才认为 residual patch 值得启用。

gate 不作为自由参数直接学，第一版用 detached teacher-side score：

```text
α_t^* = stopgrad(sigmoid(κ(normalize(m_t) - τ_m)))
```

最终 teacher target：

```text
z_t^* = z_t^g + α_t^*(z_t^{g+r} - z_t^g)
```

若 residual 无效，则 `α_t^* ≈ 0`，目标自动退回 global-only teacher。

### Step 5：蒸馏回 student

训练时 student 仍然只看 plain task prompt / normal rollout history，不看 skill。

```text
student input: q, h_t
teacher input: q, g, r_t or NULL_RES, h_t
```

loss 只作用在 student action token range 上，不作用在 environment feedback token 上。

```text
L = weighted reverse-KL(π_S(. | q, h_t), p_t^*)
```

核心目标：

```text
student 自己 rollout；
teacher 用 global skill 给所有 step 提供任务级策略监督；
只在少量必要 step 用 residual skill 修正 teacher；
再把这种修正后的 teacher 分布蒸馏回 student。
```

## 3. 记号定义

| 符号 | 含义 |
| --- | --- |
| `q` | 任务 query / 用户指令 |
| `τ` | student rollout trajectory |
| `t` | step 索引 |
| `h_t` | 第 `t` 个 step 前的历史前缀 / 状态摘要 / 交互上下文 |
| `y_t` | student 在第 `t` 步生成的动作 token 序列 |
| `g` | global skill |
| `r_t` | 第 `t` 步 residual step skill |
| `π_S` | student policy |
| `π_T` | teacher policy |
| `p_t^g` | teacher 在 `(q, g, h_t)` 下对 `y_t` 的分布 |
| `p_t^{g+r}` | teacher 在 `(q, g, r_t, h_t)` 下对 `y_t` 的分布 |
| `z_t^g, z_t^{g+r}` | 对应 logits |
| `m_t` | residual benefit score |
| `α_t^*` | stop-gradient residual gate / blend target |
| `C(τ)` | 候选 residual step 集合 |
| `M` | 实际启用 residual 的 step 数，要求 `M << T` |

## 4. 核心设计原则

### 3.1 不预设 step skill 必要

方法必须允许一种情况：

- `global skill` 已经足够；
- residual 不带来实质 teacher shift；
- 因而 `α_t^* ≈ 0`。

这不是失败，而是你要的正确行为。

### 3.2 residual 不是第二份完整 skill

`r_t` 的语义是：

> 在已经有 `g` 的条件下，这个状态还缺失的那一小块局部修正。

### 3.3 teacher 只在少数关键 step 上做 residual rescore

第一版必须把 teacher 额外开销控制在：

- 1 次 global teacher pass；
- `M` 次 residual teacher rescore；
- 其中 `M << T`。

## 5. 模块结构图

```text
Student rollout τ
    ->
Trajectory segmentation / step typing
    ->
Global skill retrieval: q -> g
    ->
Global teacher scoring: (q, g, NULL_RES, full trace) -> per-step global logits
    ->
Candidate step selection: τ, step_type, entropy -> C(τ)
    ->
For each t in C(τ):
    (q, h_t, g) -> top-k residual candidates
    ->
    summarize(q, h_t, g, candidates) -> r_t
    ->
    residual teacher scoring: (q, g, r_t, h_t) -> z_t^{g+r}
    ->
    benefit m_t, gate α_t^*
    ->
    residual target z_t^*
    ->
Token-level distillation to student
```

## 6. skill 表示设计

### 5.1 Global skill `g`

第一版不建议直接学 latent skill code，而建议先用“结构化文本 skill”：

```text
[SKILL_NAME]
[TASK_FAMILY]
[LONG_HORIZON_PLAN]
[TOOL_PATTERNS]
[FAILURE_MODES]
[VERIFICATION_CHECKLIST]
```

推荐原因：

- 易检索；
- 易人工审查；
- 与 Skill-SD 的 natural-language skill 连续；
- 后续可平滑升级成 schema + embedding。

### 5.2 Residual skill `r_t`

建议设计成短 patch 模板，而不是自由长文本：

```text
[TRIGGER]
[WHY_GLOBAL_IS_INSUFFICIENT]
[DELTA_TO_PLAN]
[LOCAL_CONSTRAINT]
[ACTION_HINT]
[EXIT_CONDITION]
```

长度预算建议远小于 global skill，例如：

- `|g|` 上限：256-512 tokens
- `|r_t|` 上限：64-128 tokens

### 5.3 为什么推荐结构化 residual

因为 residual 的目标不是“再讲一遍整个任务怎么做”，而是：

- 明确触发条件；
- 明确 delta；
- 明确何时退出。

这能强约束 residual 不退化为第二份 global skill。

## 7. teacher 输入格式

### 6.1 固定槽位模板

推荐 teacher 输入采用固定槽位：

\[
\text{TeacherInput}_t = [q] + [g] + [r_t] + [h_t]
\]

其中：

- 若没有 residual，则 `r_t = NULL_RES`
- 不累积历史 residual，不做 `r_1 + r_2 + ... + r_{t-1}`

### 6.2 为什么不能把历史 step skill 一直堆进 prompt

因为那会导致：

1. context 膨胀；
2. 位置编码与注意力模式持续漂移；
3. teacher/student mismatch 变重；
4. 后续 KV cache 复用很难做。

### 6.3 计算实现建议

为了满足 `1 + M`：

- 用一次 `global teacher pass` 在 `NULL_RES` 模板下对整条 student trace 打分，得到所有 step 的 `z_t^g`。
- 对候选 step 再单独做 `residual rescore`，只重算 `(q, g, r_t, h_t)` 到该步动作 span 的 logits。

## 8. global / residual retrieval

### 7.1 Global skill retrieval

\[
g = \text{RetrieveGlobal}(q)
\]

推荐检索键：

- query embedding
- task family
- 相关工具集合
- 成功轨迹元数据

### 7.2 Residual skill retrieval

不推荐直接：

\[
(q, h_t) \to r_t
\]

而推荐：

\[
(q, h_t, g) \to \{c_t^{(1)}, \dots, c_t^{(K)}\}
\]

再做：

\[
r_t = \text{SummarizeResidual}(q, h_t, g, \{c_t^{(k)}\}_{k=1}^K)
\]

### 7.3 为什么 “candidate retrieval + residual summarization” 更合理

因为 residual patch 是高度局部、状态依赖且容易噪声化的：

- 直接取单条 step skill，容易取到错误或过拟合样本；
- top-k candidates 能覆盖多个相似局部模式；
- summarization 可以显式去重、消冲突、并强调 “只输出相对 g 的 delta”。

### 7.4 与 OEL / OPCD / Skill-SD 的关系

- 与 `OEL` 的关系：借鉴其“先抽可迁移经验，再 consolidation”的思想，但把经验库细化成 global bank 与 residual bank。
- 与 `OPCD` 的关系：依然是 context-conditioned teacher，只是 context 从 generic 变成 hierarchical skill context。
- 与 `Skill-SD` 的关系：保留 trajectory-derived global skill 的主线，但新增 sparse state-dependent residual patch。

## 9. 候选 step 选择

### 8.1 两层机制

第一层只做筛选，不做最终判定：

\[
C(\tau) = \text{SelectCandidates}(\tau)
\]

推荐依据：

- 高 entropy step
- 或关键 step type：
  - `tool_call`
  - `replan`
  - `verify`
  - `error_recovery`

### 8.2 为什么不能直接 fixed entropy threshold

因为 entropy 在不同任务、不同模板、不同 token 类型之间不校准。  
更稳的是：

- 每轨迹分位数；
- 每 step-type 内分位数；
- 或先按关键 step type 进入候选，再做 teacher-side benefit 判定。

## 10. residual benefit 设计

对候选 step，先算：

\[
p_t^g = \pi_T(\cdot \mid q, g, h_t)
\]

\[
p_t^{g+r} = \pi_T(\cdot \mid q, g, r_t, h_t)
\]

然后定义：

\[
m_t = [ H(p_t^g) - H(p_t^{g+r}) ]_+ + \rho \cdot JS(p_t^{g+r} \| p_t^g)
\]

解释：

- 第一项：residual 让 teacher 更确定；
- 第二项：residual 真的改变了 teacher 分布，而不是只做无意义重复。

### 9.1 重要提醒

`m_t` 不是“真实 usefulness”。  
它只是在没有外部 reward / verifier 的前提下，一个 teacher-side marginal benefit proxy。

## 11. gate 设计

### 10.1 推荐 v1：teacher-side detached soft gate

先对候选步的 `m_t` 做归一化，建议用每轨迹排名或 z-score：

\[
\tilde m_t = \text{NormalizeWithinTrajectory}(m_t)
\]

再映射成：

\[
\alpha_t^* = \text{stopgrad}\left(\sigma(\kappa(\tilde m_t - \tau_m))\right)
\]

并叠加 budget：

\[
b_t = \mathbf{1}[t \in \text{TopM}(m)]
\]

\[
\alpha_t^* \leftarrow b_t \cdot \alpha_t^*
\]

### 10.2 推荐的第一版实现

我建议第一版使用：

- `candidate screening`
- `teacher-side m_t`
- `per-trajectory top-M`
- `stopgrad soft alpha`

不要一开始就做自由可学习 `α_t`。

### 10.3 可替代 gate

#### A. Hard top-M gate

- `α_t^* ∈ {0,1}`
- 优点：最省事、最易控预算
- 缺点：target 不够平滑

#### B. Soft stop-grad gate

- 当前推荐
- 优点：平滑、稳定、不会被 student loss 拉塌

#### C. Learnable gate with separate supervision

若以后要学 gate，建议：

1. 用 `m_t` 或 top-M 标签先生成 pseudo labels
2. 训练一个 `gate predictor f_\psi(q, h_t, g)` 去拟合这些 labels
3. 训练时只把其输出 stop-gradient 后作为调度/预算信号

而不是直接让 `α_t` 吃 KL loss。

## 12. residual target 构造

推荐目标：

\[
z_t^* = z_t^g + \alpha_t^*(z_t^{g+r} - z_t^g)
\]

然后：

\[
\tilde p_t = \text{softmax}(z_t^*)
\]

### 11.1 为什么不用直接 `p_t^{g+r}`

因为你的核心问题不是：

- “让 student 学会 teacher 在 skill prompt 下的完整分布”

而是：

- “让 student 学会 relative to global teacher 的局部修正”

差分 target 更符合 residual patch 的语义，也更方便分析 residual 到底改了什么。

### 11.2 等价解释

logit 残差插值等价于在概率空间做几何插值：

\[
\tilde p_t \propto (p_t^g)^{1-\alpha_t^*} (p_t^{g+r})^{\alpha_t^*}
\]

这说明它并不是粗暴线性平均，而是保留了 “以 global teacher 为底座、向 residual teacher 方向偏移” 的解释。

## 13. 训练目标

student 始终只看 plain task prompt，不看 `g` 和 `r_t`。  
teacher 才接收 skill context。

建议总损失写成：

\[
\mathcal{L} = \sum_t \lambda_t \, D_{\text{revKL}}\left(\pi_S(\cdot \mid q, h_t)\; || \; \tilde p_t \right)
\]

其中：

- 对非候选步，`α_t^*=0`，即 `\tilde p_t = p_t^g`
- 对激活 residual 的步，`\tilde p_t = softmax(z_t^*)`
- `\lambda_t` 可沿用 Skill-SD 的 importance weighting / token weighting 思路

### 12.1 推荐继承而不是重造的部分

建议直接继承 Skill-SD / OPCD 里已经验证过的：

- student-owned rollout
- reverse-KL / weighted reverse-KL
- teacher periodic synchronization

真正新增的是：

- hierarchical skill retrieval
- residual-only-on-key-steps
- teacher-side benefit gate
- residual target construction

## 14. 离线 teacher-scoring prototype

给定一批 student trajectories，prototype 要做的是：

1. `q -> g`
2. 一次 global teacher scoring
3. 选候选 step
4. 对候选 step 检索 residual candidates
5. summarize 成 `r_t`
6. 做 residual teacher scoring
7. 计算 `m_t`
8. 计算 `α_t^*`
9. 构造 `z_t^*`
10. 输出统计量与可视化

### 13.1 建议统计量

- 每轨迹候选 step 数
- 每轨迹激活 residual step 数
- `m_t` 直方图
- `α_t^*` 直方图
- 按 step type 的 entropy drop / JS / benefit
- residual 是否改变 top-1 / top-k action
- residual target 对 student KL 的改善量
- residual prompt token 开销

## 15. 合理性、可行性、风险

### 14.1 合理性

方法的最强处在于它不强行假定 step skill 总是有用，而是把这个问题转成可测的 teacher-side marginal correction。

### 14.2 可行性

工程上它是可行的，因为：

- `OPCD/OEL` 已经证明了 context-conditioned on-policy distillation 的训练链路；
- `G-OPD` 已经给出了 ref-input 改造的清晰代码落点；
- `VeRL` 已经提供了实际可用的 rollout / trainer / multi-turn infrastructure。

### 14.3 主要风险

- residual summarization 可能退化成 second global skill
- candidate screening 可能漏掉重要低熵 step
- teacher-side benefit 与真实任务收益不总一致
- 额外 teacher rescore 仍然昂贵

## 16. 第一版我建议坚持的边界

第一版不要做：

- 多 residual mixture
- learnable free gate
- counterfactual step recovery
- AppWorld full benchmark
- “真实 usefulness” 强 claim

第一版要做好：

- global / residual 的层次分工
- sparse gating
- residual target
- 离线 teacher-side 证据链

## Sources

- Skill-SD: [arXiv:2604.10674](https://arxiv.org/abs/2604.10674)
- OPCD: [arXiv:2602.12275](https://arxiv.org/abs/2602.12275)
- OEL: [arXiv:2603.16856](https://arxiv.org/abs/2603.16856)
- G-OPD: [arXiv:2602.12125](https://arxiv.org/abs/2602.12125)
- OPSD: [arXiv:2601.18734](https://arxiv.org/abs/2601.18734)
