# Skill-SD 论文阅读文档与教程

论文：`Skill-SD: Skill-Conditioned Self-Distillation for Multi-turn LLM Agents`

版本：arXiv:2604.10674 v1，2026-04-12

官方页面：

- Paper: https://arxiv.org/abs/2604.10674
- PDF: https://arxiv.org/pdf/2604.10674
- Project page: https://skill-sd.github.io/

本地资源：

- `D:\codex_workspace\skill_opd\resource\papers\2604.10674_Skill-SD.pdf`
- `D:\codex_workspace\skill_opd\resource\web\skill-sd_project.html`
- 抽取文本：`D:\codex_workspace\skill_opd\resource\_analysis\raw\skill_sd_pdf_text.txt`

## 1. 一句话读懂

Skill-SD 的核心不是“给 student 加 skill prompt”，而是：

```text
student 始终只看普通任务 prompt 做 on-policy rollout；
历史 trajectory 被总结成 natural-language skill；
skill 只给 teacher 看；
teacher 在 student 自己采样出来的 token 上重新打分；
student 通过 importance-weighted reverse-KL 把 teacher 的局部 token-level 偏好学进去。
```

更短地说：

```text
Skill-SD = GRPO 的任务级奖励 + teacher-only skill 的 token 级蒸馏
```

论文最重要的工程结论：

```text
skills should guide the teacher, not the student
```

即 skill 最好作为训练时 teacher 的 privileged information，而不是直接拼进 student rollout prompt。

## 2. 这篇论文到底解决什么问题

多轮 LLM agent 的 RL 训练有两个典型问题：

```text
1. reward 稀疏：
   agent 可能走了 20 步，只在最后拿到成功/失败信号。

2. horizon 长：
   哪一个 token / action 真正导致成功或失败，很难靠最终 reward 反推。
```

OPD / OPSD 的思路是：

```text
student 自己 rollout；
teacher 对 student 生成的 token 重新打分；
用 teacher 的 token-level signal 给 student 更密集的监督。
```

但在 multi-turn agent 里，传统 OPSD 的 privileged information 往往是固定答案、ground-truth solution、或某种外部 context。论文认为这不适合 agent task，因为：

```text
agent task 通常没有唯一正确轨迹；
AppWorld 同一个任务可以用多种 API 顺序解决；
Sokoban 也可能存在多条可行路径；
固定答案会过度约束探索。
```

Skill-SD 的回答是：

```text
不要用固定答案当 teacher privilege；
用 agent 自己历史 trajectory 总结出的 skill 当 teacher privilege。
```

## 3. 方法总流程

可以按 4 步理解。

### 3.1 Step 1：student plain-prompt on-policy rollout

student 只看普通任务 prompt：

```text
π_stu(a_t | x, y_<t)
```

这里：

```text
x      = 当前任务 prompt
y_<t   = 当前 step 前已经生成/执行过的 action token history
π_stu  = student policy
```

关键点：

```text
student 不看 skill；
训练时和测试时 student 输入一致；
避免 train-test prompt mismatch。
```

这点对我们当前项目非常重要。我们现在做 Sokoban offline rollout exporter，导出的就是这种 student-owned plain-prompt trajectory。

### 3.2 Step 2：trajectory-to-skill

完成一次 trajectory 后，论文用辅助 LLM 把完整轨迹总结成一个 skill：

```text
e = Analyze(τ, x)
  = (e_success, e_mistake, e_workflow)
```

论文附录里的 skill 是一个 JSON 风格结构，包含三类内容：

```text
success_analysis:
  哪些策略是对的，哪些工具/动作选择是合理的。

mistake_analysis:
  失败来自哪里，比如错误假设、没有验证 API 参数、重复认证失败等。

golden_workflow:
  下次遇到类似任务时应该遵循的高层流程。
```

注意：

```text
Skill-SD 不存完整 trajectory 作为 skill；
它把 trajectory 压缩成 compact natural-language skill。
```

这样做的动机是：

```text
完整轨迹太长；
多轮任务可能有多种可行解；
照抄单条轨迹会过度约束 agent 探索。
```

### 3.3 Step 3：skill retrieval

对每个任务 `x`，Skill-SD 从 skill bank 里选一个 skill：

```text
S(x) = UCB-retrieve(B(x))
```

论文使用轻量 UCB，而不是 embedding retrieval：

```text
score(e) = mean_reward(e) + c * sqrt(ln N_ucb / n(e))
```

含义：

```text
mean_reward(e):
  这个 skill 历史上带来的平均 reward。

n(e):
  这个 skill 被选过多少次。

N_ucb:
  当前 task 下总 retrieval 次数。

c:
  exploration/exploitation tradeoff。
```

论文还规定：

```text
从未被选过的新 skill 先被尝试。
```

这让 skill bank 不会只用早期高分 skill，也会探索新总结出来的 skill。

### 3.4 Step 4：teacher-only skill replay

teacher 看的是：

```text
x + S(x) + y_<t
```

student 看的是：

```text
x + y_<t
```

公式上：

```text
π_stu_θ(. | x, y_<t) = π_θ(. | x, y_<t)

π_tea_θbar(. | x, S(x), y_<t) = π_θbar(. | x ⊕ S(x), y_<t)
```

注意两个关键点：

```text
1. teacher 和 student 可以是同一个模型族/同一套参数的不同视角。
2. 区别主要来自 prompt conditioning：teacher 多看 skill。
```

Skill-SD 不是让 teacher 自己 rollout，而是：

```text
student rollout 得到 token 序列；
teacher 对同一条 student token 序列重新计算 log-prob。
```

这叫 student-owned on-policy rollout。

## 4. 损失函数怎么理解

Skill-SD 的总目标：

```text
L_total = L_GRPO + λ L_SDL
```

其中：

```text
L_GRPO:
  任务级 reward learning，决定哪些 trajectory 应该强化。

L_SDL:
  teacher-only skill 的 token-level distillation，决定同一条 trajectory 里哪些 token 更像 skill-conditioned teacher。
```

### 4.1 GRPO 部分

对同一个任务采样 `G` 条 trajectory：

```text
τ_i ~ π_stu_old(. | x)
```

每条 trajectory 得到 reward：

```text
R_i = R(x, τ_i)
```

组内标准化 advantage：

```text
A_i = (R_i - mean(R_1...R_G)) / (std(R_1...R_G) + ε)
```

直觉：

```text
同一个任务下，比同组其他 rollout 更好的 trajectory 得正 advantage；
更差的 trajectory 得负 advantage。
```

这提供任务级方向，但不能精确告诉每个 token 应该怎么改。

### 4.2 SDL 部分

论文的 self-distillation loss 是 sampled-token reverse-KL，而不是 full-vocab KD。

对 student 已经采样出的 token `y_i,t`，重新计算：

```text
log π_stu_θ(y_i,t | x, y_i,<t)
log π_tea_θbar(y_i,t | x, S(x), y_i,<t)
```

定义 student-teacher log-ratio：

```text
ℓ_i,t = log π_stu_θ(y_i,t | x, y_i,<t)
      - log π_tea_θbar(y_i,t | x, S(x), y_i,<t)
```

再定义 on-policy importance weight：

```text
ρ_on_i,t = π_stu_θ(y_i,t | x, y_i,<t)
           / stopgrad(π_stu_old(y_i,t | x, y_i,<t))
```

SDL loss：

```text
L_SDL = mean over valid tokens [
  ρ_on_i,t * ( exp(-ℓ_i,t) - 1 + ℓ_i,t )
]
```

这个式子可以简化理解为：

```text
如果 teacher 比 student 更喜欢当前 token，
则推动 student 提高该 token 概率；

如果 teacher 和 student 已经一致，
SDL 梯度自然变小；

如果 student 当前分布和 rollout old policy 有偏移，
importance weight 负责修正采样分布不一致的问题。
```

论文强调 importance weight 的原因：

```text
直接对 naive reverse-KL / k3 estimator 求梯度会有 per-token gradient bias；
在 cross-prompt setting 里，teacher 和 student prompt 不同，这个问题更明显；
所以需要 importance correction。
```

### 4.3 为什么 λ 要小

论文把 SDL 视为辅助 shaping signal，不是主学习目标。

原因：

```text
GRPO 负责 reward-grounded learning；
SDL 只是把 teacher-side skill 信息变成 token-level guidance。
```

如果 `λ` 太大：

```text
student 会过度贴近 teacher；
可能压制探索；
甚至和 reward-improving direction 冲突。
```

论文实验里 AppWorld 上：

```text
λ = 0.001 效果最好；
λ = 0.01 过强，过度 regularize；
λ = 0.0005 太弱，teacher guidance 不够。
```

## 5. 为什么 skill 不能直接给 student

这是论文最值得记住的一点。

一个看起来很自然的 baseline 是：

```text
Skill-Augmented GRPO:
  直接把 skill 拼到 student rollout prompt 里；
  然后照常做 GRPO。
```

实验结果说明这很差：

```text
AppWorld:
  Vanilla GRPO: 50.9 Acc.
  Skill-Augmented GRPO: 42.1 Acc.
  Skill-SD: 64.9 Acc.

Sokoban:
  Vanilla GRPO: 51.6 Acc.
  Skill-Augmented GRPO: 20.3 Acc.
  Skill-SD: 62.5 Acc.
```

论文解释：

```text
训练时 student policy 是 π(a | h, skill)；
测试时 student policy 是 π(a | h)；
这两个 conditional policy 共享参数，但不是同一个策略。
```

所以直接 skill prompting 会造成：

```text
train-test mismatch；
student 依赖 retrieval；
prompt 变长；
规划能力被 skill token 干扰；
在 Sokoban 这种不可逆任务上尤其严重。
```

Skill-SD 的处理：

```text
skill 只给 teacher；
student 的输入始终干净；
student 通过梯度把有用的 skill 信息内化进参数。
```

## 6. 为什么必须 student-owned rollout

论文比较了四种设置：

```text
1. On-policy + Dynamic teacher   = Skill-SD
2. On-policy + Frozen teacher
3. Off-policy teacher rollout + Dynamic teacher
4. Off-policy teacher rollout + Frozen teacher
```

结论：

```text
student-owned on-policy rollout 是稳定训练的必要条件；
dynamic teacher synchronization 是进一步提升性能的关键。
```

论文的 ablation 结果：

```text
On-policy + Dynamic:
  AppWorld 64.9 Acc.
  Sokoban 62.5 Acc.

On-policy + Frozen:
  AppWorld 49.1 Acc.
  Sokoban 50.0 Acc.

Off-policy + Frozen:
  AppWorld 45.6 Acc.
  Sokoban 12.5 Acc.

Off-policy + Dynamic:
  AppWorld 42.1 Acc.
  Sokoban 10.9 Acc.
```

为什么 off-policy 会崩：

```text
如果 trajectory 是 teacher 采样的，
student 训练时要追 teacher 的分布；
随着 student 和 teacher rollout distribution 分离，
importance ratio 变得不稳定；
训练中期可能 collapse。
```

Sokoban 崩得更严重，因为：

```text
Sokoban 是高不可逆环境；
一个错误推箱动作可能直接让整局无解；
轻微 distribution mismatch 也会导致 catastrophic failure。
```

这也解释了为什么我们当前项目第一步必须先做：

```text
student trajectory generation / offline rollout export
```

而不是先写 teacher scoring。

## 7. dynamic teacher synchronization 怎么理解

Skill-SD 的 teacher 不是永远 frozen 的旧模型。

每个 iteration：

```text
θ_old ← θ
teacher θbar ← current student θ
student 用 θ_old rollout
teacher 用 θbar + skill 重新打分
student 更新 θ
```

直觉：

```text
teacher 要比 student 多看 skill；
但 teacher 的基础能力不能长期停留在早期 checkpoint；
否则 teacher signal 会越来越陈旧。
```

dynamic teacher 的好处：

```text
teacher 的能力随 student 改进而更新；
teacher-student gap 主要来自 skill context，而不是模型能力年代差；
distillation signal 更 calibrated。
```

frozen teacher 的问题：

```text
稳定，但上限低；
student 后期可能超过 teacher；
teacher 变成旧策略 regularizer。
```

## 8. 实验怎么读

### 8.1 Benchmark

论文使用两个互补环境：

```text
AppWorld:
  多应用 API 操作；
  错误相对可恢复；
  需要 API coordination、state management、replanning。

Sokoban:
  空间规划；
  long-horizon；
  动作高度不可逆；
  错误推箱可能直接失败。
```

这两个环境组合很合理：

```text
AppWorld 测工具/API agent；
Sokoban 测规划/不可逆决策。
```

### 8.2 训练和评测设置

论文设置：

```text
Base model:
  Qwen3-4B-Instruct-2507

AppWorld:
  train: 90 public tasks
  eval: 57 dev tasks
  max turns: H = 40

Sokoban:
  6x6 room
  2 boxes
  train: 96 generated levels with curriculum
  test: 64 levels
  max steps: H = 40
```

指标：

```text
Accuracy / pass@1:
  是否完整成功。

Completion rate:
  dense reward / partial completion。
```

### 8.3 主表应该怎么读

主结果：

| Method | AppWorld Acc. | AppWorld Comp. | Sokoban Acc. | Sokoban Comp. | Avg. Acc. | Avg. Comp. |
|---|---:|---:|---:|---:|---:|---:|
| Base Model | 8.8 | 39.1 | 12.5 | 32.0 | 10.6 | 35.6 |
| Vanilla OPD | 22.8 | 59.7 | 21.9 | 37.5 | 22.4 | 48.6 |
| Vanilla GRPO | 50.9 | 76.3 | 51.6 | 68.8 | 51.2 | 72.5 |
| Skill-Augmented GRPO | 42.1 | 76.1 | 20.3 | 37.5 | 31.2 | 56.8 |
| Skill-SD | 64.9 | 84.9 | 62.5 | 71.1 | 63.7 | 78.0 |

应读出的结论：

```text
1. RL reward 很重要：
   Vanilla GRPO 远强于 Vanilla OPD。

2. 直接 skill prompting 不可靠：
   Skill-Augmented GRPO 甚至低于 Vanilla GRPO。

3. teacher-only skill distillation 有效：
   Skill-SD 是最强方法。

4. Sokoban 上直接加 skill prompt 特别差：
   20.3 vs Vanilla GRPO 51.6，说明 skill token 会严重干扰规划。
```

## 9. 论文的真正贡献

我建议把贡献理解成 4 个层次。

### 9.1 任务定义贡献

它指出 multi-turn agent 的 privileged information 不能简单等同于 ground-truth answer。

原因：

```text
agent 任务有多条可行轨迹；
固定答案不适合作为 teacher context。
```

### 9.2 skill 作为动态 teacher context

它把 trajectory 总结成 skill，并作为 teacher-only privileged context。

这比普通 memory / retrieval 更严格，因为：

```text
skill 不改变 student rollout policy；
skill 只改变 teacher scoring distribution；
student 最终必须内化到参数里。
```

### 9.3 importance-weighted reverse-KL

论文不是简单做 KL/student-teacher CE。

它强调：

```text
student trajectory 来自 old policy；
current student 和 teacher prompt 不同；
直接求 reverse-KL 梯度有偏；
需要 on-policy importance correction。
```

这是它比“prompt teacher + distill”更严肃的地方。

### 9.4 dynamic teacher synchronization

它验证：

```text
teacher 不能长期 frozen；
teacher-owned rollout 不稳定；
on-policy student rollout + dynamic teacher 是关键组合。
```

## 10. 论文的限制

论文自己承认几个限制。

### 10.1 Retrieval 很简单

Skill-SD 用 per-task UCB retrieval，不是 embedding retrieval。

优点：

```text
简单；
不引入额外 retrieval model；
统计意义清楚。
```

缺点：

```text
难以跨任务泛化；
skill bank 变大后可能选得不够语义化；
不能处理状态依赖的 step-level skill。
```

### 10.2 只做 sampled-token distillation

它只在 student 已采样 token 上比较 teacher/student log-prob，不做 full-vocab KD。

优点：

```text
省显存；
适合 long-horizon agent trace。
```

缺点：

```text
无法完整看到 teacher distribution 的全量变化；
对我们想比较 global teacher vs global+residual teacher 的分布差异来说不够。
```

### 10.3 skill 还是 global natural-language summary

Skill-SD 的 skill 是整条 trajectory 的 summary。

它没有解决：

```text
global skill 和 step skill 的层次关系；
某一步是否真的需要 skill；
如何避免每一步都拼 skill；
状态依赖的 residual correction；
global-only teacher 与 global+residual teacher 的差异建模。
```

这正是我们 Hierarchical Residual Skill OPD 要推进的地方。

## 11. 和我们方法的关系

我们的目标不是否定 Skill-SD，而是继承它最有效的部分，再补它没有处理的层次结构。

### 11.1 我们应该继承的部分

必须继承：

```text
1. student-owned rollout
2. teacher-only skill conditioning
3. student plain prompt at train/test
4. skill as trajectory-derived privileged information
5. reward + distillation 双信号
6. teacher synchronization / dynamic teacher 思路
```

尤其是：

```text
不要把 skill 直接拼进 student prompt。
```

### 11.2 我们要改进的部分

Skill-SD：

```text
teacher sees: x + global skill + h_t
```

我们的方法：

```text
global-only teacher:
  p_t^g = π_T(. | q, g, h_t)

global+residual teacher:
  p_t^{g+r} = π_T(. | q, g, r_t, h_t)

residual target:
  z_t* = z_t^g + α_t* (z_t^{g+r} - z_t^g)
```

也就是说：

```text
Skill-SD 证明 global skill 给 teacher 有用；
我们要回答什么时候 global skill 不够，以及局部 residual patch 是否值得启用。
```

### 11.3 我们和 Skill-SD 的关键区别

| 维度 | Skill-SD | Hierarchical Residual Skill OPD |
|---|---|---|
| skill 粒度 | trajectory-level global summary | global skill + sparse step residual |
| step skill | 没有显式建模 | 只在关键 step 上出现 |
| teacher 条件 | task + skill | task + global skill + optional residual |
| residual 思路 | 无 | 用 global+residual 相对 global-only 的 teacher shift |
| gate | 无 | teacher-side benefit score -> α_t* |
| 主要问题 | skill 是否整体有用 | 当前状态下是否需要 residual patch |

一句话：

```text
Skill-SD 是 global teacher-only skill distillation；
我们要做 hierarchical residual teacher-only skill distillation。
```

## 12. 阅读路线教程

如果你只有 20 分钟：

```text
1. 读 Abstract 和 Introduction。
2. 看 Figure 1。
3. 读 Section 3.2。
4. 看 Table 1。
5. 读 Section 4.3 里关于 student-owned rollout 和 teacher-only skill 的 ablation。
```

如果你有 60 分钟：

```text
1. 先按 20 分钟路线读。
2. 读 Section 3.1，理解 GRPO backbone。
3. 读 Section 3.3，重点理解 ℓ_i,t 和 ρ_on_i,t。
4. 读 Algorithm 1，把每一步和训练流程对应起来。
5. 读 Appendix A 的 skill format。
```

如果你要复现或改进：

```text
1. 先实现 student rollout export。
2. 再实现 trajectory-to-skill summarization。
3. 再实现 skill bank 和 retrieval。
4. 再实现 teacher re-scoring。
5. 最后才改 trainer loss。
```

不要反过来做。原因是：

```text
没有可靠 student trajectory，就没有 skill；
没有 skill，就没有 teacher-only context；
没有 teacher re-scoring，就不应该改 loss。
```

## 13. 给工程实现的伪代码

Skill-SD 可以写成下面这种结构：

```python
for iteration in range(num_iters):
    # 1. dynamic teacher sync
    old_student = copy_current_policy(student)
    teacher = sync_from_student(student)

    # 2. rollout, student sees plain prompt only
    trajectories = []
    for task in batch_tasks:
        skill = skill_bank.retrieve(task)  # may be empty
        rollouts = rollout_with_student(
            policy=old_student,
            task_prompt=task.prompt,
            skill=None,
            group_size=G,
        )
        trajectories.extend(rollouts)

    # 3. rewards and GRPO advantages
    rewards = env_reward(trajectories)
    advantages = group_normalize(rewards)

    # 4. re-score same tokens
    for traj in trajectories:
        student_logp = current_student_logprob(
            prompt=traj.task_prompt,
            tokens=traj.tokens,
        )
        teacher_logp = teacher_logprob(
            prompt=traj.task_prompt,
            skill=skill_bank.retrieve(traj.task),
            tokens=traj.tokens,
        )
        old_logp = traj.old_student_logp

    # 5. optimize
    loss = grpo_loss(student_logp, old_logp, advantages)
    loss += lambda_sdl * skill_sd_loss(student_logp, teacher_logp, old_logp)
    update(student, loss)

    # 6. async skill update for future iterations
    async_summarize_and_insert(trajectories, skill_bank)
```

## 14. 映射到当前 verl-agent-opd 项目

当前项目已经完成的是 Skill-SD 复现/扩展路线里的第一步：

```text
student plain-prompt rollout export
```

当前代码对应：

```text
agent_system/multi_turn_rollout/rollout_loop.py
agent_system/skill_opd/rollout_hook.py
agent_system/skill_opd/rollout_exporter.py
examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

下一步不要直接写 loss，而应该做：

```text
1. 在服务器跑出 Sokoban JSONL。
2. 确认 prompt_text 是否足够表示 h_t。
3. 写 trajectory-to-global-skill summarizer。
4. 写 skill bank schema。
5. 写 offline teacher scoring prototype。
6. 再讨论如何把 teacher score 接入 trainer。
```

## 15. 读完后应该能回答的问题

你读完这篇论文后，至少应该能回答：

```text
1. 为什么 skill 不直接给 student？
2. 为什么 trajectory 必须由 student 自己 rollout？
3. teacher-only skill conditioning 和 ordinary prompt engineering 有什么区别？
4. Skill-SD 的 skill 是什么格式？
5. GRPO 和 SDL 分别解决什么问题？
6. importance-weighted reverse-KL 为什么需要 importance weight？
7. dynamic teacher 为什么比 frozen teacher 好？
8. 为什么 Vanilla OPD 不够？
9. 为什么 Skill-Augmented GRPO 会比 Vanilla GRPO 差？
10. 我们的 residual step skill 相比 Skill-SD 具体推进了什么？
```

## 16. 最重要的 takeaway

这篇论文对我们项目最重要的不是公式本身，而是三个设计原则：

```text
1. student rollout 必须保持 plain prompt on-policy；
2. skill 应该作为 teacher-side privileged information；
3. skill distillation 必须受 reward-grounded RL 约束，不能单独漂移。
```

我们的 Hierarchical Residual Skill OPD 应该严格继承这三点，然后扩展：

```text
global skill 负责长期任务先验；
step residual skill 只在 global skill 不足以解释局部决策时启用；
residual 的价值由 teacher-side distribution shift / benefit score 判断，而不是预设每一步都需要 skill。
```

