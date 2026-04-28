# Active Backlog

## Immediate

- Run `examples/skill_opd/run_sokoban_rollout_export_qwen3.sh` on the GPU
  server with `Qwen/Qwen3-4B`.
- Inspect the generated JSONL and confirm real rollout fields.
- If real rollout records lack exact terminal `dones`, add a minimal `dones`
  field to the exporter input from `vanilla_multi_turn_loop`.
- If response masks are needed for offline distillation batches, decide whether
  to export `attention_mask`, `responses`, and response mask from vLLM outputs.
- Keep generated rollout files out of Git.

## Short Term

- Add a small `inspect_rollout_jsonl.py` script for trajectory statistics.
- Add a converter from raw rollout JSONL to skill memory JSONL.
- Define the Sokoban skill memory schema.
- Add a few manually reviewed example trajectories to docs, not large data.
- Compare SkillRL memory format and decide what to reuse.

## Method Prototype

- Implement candidate step selection on exported trajectories.
- Implement global skill retrieval stub.
- Implement residual skill retrieval stub.
- Implement residual summarization stub.
- Implement fake-logits unit tests for residual benefit and gate computation.

## Research Questions

- Does Sokoban require residual step skills beyond global skill?
- Which step types should enter the candidate set: invalid action, high entropy,
  wall/box interaction, repeated state, or final verification?
- Should the first prototype use teacher entropy only, or entropy plus JS shift?
- How many residual teacher passes per trajectory are affordable?

## Blockers

- Need one successful real GPU rollout export to verify actual verl-agent fields.
- Need clarity on whether Qwen3 text-only model is sufficient for Sokoban text
  observation mode, or whether visual mode should remain Qwen2.5-VL first.
- Need server-side vLLM/Ray compatibility check with the existing conda env.

