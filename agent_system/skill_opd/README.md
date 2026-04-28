# Skill-OPD Phase 1 Rollout Export

This package adds a minimal offline rollout export path to verl-agent.

Current scope:

- Reuse verl-agent Sokoban environment, prompts, rollout loop, tokenizer, and generation code.
- Export completed student trajectories as JSONL after each `vanilla_multi_turn_loop`.
- Keep the exported schema stable enough for later global skill retrieval, residual skill retrieval, and teacher scoring.

Non-goals in phase 1:

- No skill retrieval.
- No teacher scoring.
- No distillation loss.
- No new environment implementation.

Enable export with Hydra overrides:

```bash
+skill_opd.export_rollouts=True \
+skill_opd.export_path=outputs/skill_opd/rollouts/sokoban_qwen3.jsonl \
+skill_opd.overwrite=True
```

Each JSONL line is one trajectory.  The exporter keeps `available_keys` per
step because verl-agent rollout fields can differ across rollout backends and
config options.
