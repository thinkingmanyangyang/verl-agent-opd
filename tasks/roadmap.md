# Skill-OPD Roadmap

## Phase 1: Offline Student Rollout Export

Goal: use verl-agent's native Sokoban rollout path to generate student
trajectories as JSONL.

Status: implemented locally and pushed as initial code.

Done:

- Added `agent_system/skill_opd/` exporter modules.
- Hooked exporter after `TrajectoryCollector.vanilla_multi_turn_loop()`.
- Added Sokoban data/model/export scripts under `examples/skill_opd/`.
- Added research/design docs under `docs/skill_opd/`.
- Verified py_compile, bash syntax, and fake JSONL export on CPU.

Remaining:

- Run real GPU validation-only rollout on server.
- Inspect real JSONL fields and confirm prompt/response token availability.
- Decide whether exact `dones`, response masks, and rollout log probs must be
  exported in phase 1.1.

## Phase 2: Trajectory to Skill Memory

Goal: convert raw trajectory JSONL into reusable skill memory records.

Tasks:

- Define Sokoban memory schema with task state, action sequence, outcome, and
  failure mode.
- Implement `raw_trajectories.jsonl -> skill_memories.jsonl`.
- Keep token-level rollout fields separate from natural-language memory fields.
- Compare schema against SkillRL generated memory files.
- Add small inspection scripts for success/failure trajectory examples.

## Phase 3: Global and Residual Skill Construction

Goal: build the skill inputs needed by hierarchical residual Skill-OPD.

Tasks:

- Implement global skill retrieval from `(query)` or task family metadata.
- Implement residual candidate retrieval from `(query, h_t, global_skill)`.
- Implement residual summarization from top-k candidates.
- Keep residual skill as a local patch, not a second full trajectory skill.
- Add NULL residual placeholder for non-gated steps.

## Phase 4: Offline Teacher Scoring Prototype

Goal: compute teacher-side residual benefit without training.

Tasks:

- Build fixed-slot teacher input: `[query] + [global_skill] + [residual_skill] + [h_t]`.
- Compute global-only teacher logits `z_g`.
- Compute global+residual teacher logits `z_gr` only for candidate steps.
- Compute residual benefit:

```text
m_t = [H(p_g) - H(p_gr)]_+ + rho * JS(p_gr || p_g)
```

- Compute detached gate:

```text
alpha_t = stopgrad(sigmoid(kappa * (normalize(m_t) - tau_m)))
```

- Build residual target:

```text
z_star = z_g + alpha_t * (z_gr - z_g)
```

- Save statistics for entropy drop, JS shift, gate distribution, and cost.

## Phase 5: Distillation Integration

Goal: integrate residual teacher targets into training.

Tasks:

- Start with offline distillation batches before online trainer changes.
- Add loss only on student response tokens.
- Compare direct `global+residual` distill vs residual target distill.
- Keep student prompt skill-free.
- Avoid all-step residual scoring; enforce `M << T`.

## Phase 6: Benchmarks and Ablations

Initial benchmark priority:

- Sokoban first, because verl-agent already has the environment path.
- AppWorld second, after rollout schema is stable.
- WebShop third, if server environment setup is tractable.

Core baselines:

- No skill.
- Vanilla OPD / OPSD.
- OPCD-style context-conditioned teacher.
- Skill-SD global skill only.
- Global + always-on step skill.
- Global + gated residual step skill.
- Direct teacher+skill distill.
- Residual target distill.

