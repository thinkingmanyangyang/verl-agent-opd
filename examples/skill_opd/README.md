# Skill-OPD Phase 1 Scripts

These scripts are thin wrappers around verl-agent.  They are meant to produce
offline student rollouts for later skill extraction and teacher scoring.

Files:

- `prepare_sokoban_data.sh`: creates the small text-mode parquet files expected
  by verl-agent.  It reuses `examples.data_preprocess.prepare`.
- `download_qwen3_model.sh`: downloads `Qwen/Qwen3-4B` or `Qwen/Qwen3-8B` to a
  local Hugging Face cache directory.
- `run_sokoban_rollout_export_qwen3.sh`: runs the existing verl-agent PPO entry
  with rollout export enabled.  This requires a GPU/vLLM setup and should not be
  run on the current CPU-only machine.

The export hook writes one JSON object per trajectory to:

```text
outputs/skill_opd/rollouts/sokoban_qwen3.jsonl
```

Each step keeps prompt text, model response text, response token ids, reward,
valid-action flag, env info, and raw available keys.

