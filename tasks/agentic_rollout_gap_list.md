# Skill-OPD Agentic Rollout Gap List

Last updated: 2026-05-07

This file lists what is complete, what was minimally fixed, and what still needs
server validation for the agentic env/rollout layer.

## 1. Completed Locally

### 1.1 Rollout Export Hook

Status: implemented.

Location:

```text
agent_system/multi_turn_rollout/rollout_loop.py
agent_system/skill_opd/rollout_hook.py
agent_system/skill_opd/rollout_exporter.py
```

Behavior:

```text
success = envs.success_evaluator(...)
maybe_export_rollout(...)
return total_batch_list, episode_rewards, ...
```

Why this is the correct hook:

- `total_batch_list` and `total_infos` are complete only after rollout.
- Step-level export would duplicate the existing collection logic.
- Full trajectory export is easier to align with skill extraction and teacher
  scoring.

### 1.2 Exact Done Export

Status: minimally fixed.

Change:

```text
rollout_loop.py now stores batch.non_tensor_batch["dones"].
rollout_exporter.py reads step_data["dones"] first.
```

Fallback:

```text
If older records lack "dones", exporter falls back to the last active step and
sets metadata.done_is_inferred_from_episode_length = true.
```

### 1.3 Raw Text Action and Projected Action

Status: minimally fixed for the relevant environment managers.

Change:

```text
EnvManager.step stores:
  info["raw_text_action"]
  info["projected_action"]

rollout_loop.py stores:
  batch.non_tensor_batch["text_action"]
  batch.non_tensor_batch["projected_action"]
```

Why this matters:

- `response_text` is the full model output.
- `text_action` is the decoded model response as sent to projection.
- `projected_action` is the actual action sent to the environment.
- Skill-OPD needs all three to analyze local residual corrections.

### 1.4 Response Mask and Rollout Log Probs

Status: exporter support added.

Current behavior:

```text
response_mask:
  uses explicit step_data["response_mask"] if present
  otherwise derives attention_mask[-len(responses):]

rollout_log_probs:
  exported if step_data["rollout_log_probs"] exists
```

This avoids changing trainer loss behavior.

## 2. Not Yet Validated

### 2.1 Sokoban Real GPU Rollout

Status: required before claiming phase 1 complete.

Command:

```bash
MODEL_PATH=$HOME/models/Qwen3-4B \
DATA_ROOT=$HOME/data/verl-agent \
TRAIN_DATA_SIZE=8 \
VAL_DATA_SIZE=8 \
GROUP_SIZE=1 \
MAX_STEPS=15 \
NUM_GPUS=1 \
TP_SIZE=1 \
EXPORT_PATH=outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl \
bash examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

Validation commands:

```bash
wc -l outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl
head -n 1 outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl
```

Must verify:

```text
trajectory count >= 8
steps length >= 1
prompt_text non-empty
response_text non-empty
response_token_ids non-empty
text_action non-empty
projected_action present
done present
info contains raw_text_action/projected_action/is_action_valid
```

### 2.2 Prompt Text Quality

Risk:

`prompt_text` may decode from `prompts`, `raw_prompt_ids`, or `input_ids`.
Depending on backend fields, it might include chat-template artifacts or not
exactly match the human-readable current `h_t`.

Validation:

Inspect real JSONL:

```text
steps[0].prompt_text
steps[1].prompt_text
steps[1].info
steps[1].available_keys
```

If insufficient:

```text
Save raw current observation or memory_context explicitly in EnvManager info.
```

Do not add this before checking real output.

### 2.3 Rollout Log Probs

Risk:

`vllm_rollout_spmd.py` returns `rollout_log_probs`, but availability depends on
vLLM output logprobs and config behavior.

Validation:

Check:

```text
steps[].rollout_log_probs
steps[].available_keys
```

If missing:

```text
Do not block phase 1. Trainer recomputes old_log_probs later.
Only add mandatory export if offline distillation needs rollout policy probs.
```

## 3. Dataset-Specific Gaps

### 3.1 Sokoban

Current readiness:

```text
env package: yes
EnvManager: yes
projection: yes
history/memory: yes
success evaluator: yes
Skill-OPD exporter: yes
real GPU validation: pending
```

Potential follow-ups after real run:

```text
Add raw room/grid state to info if prompt_text is not enough.
Add step type classifier: invalid, wall hit, box push, final, success.
Add trajectory inspection script.
```

### 3.2 AppWorld

Current readiness:

```text
env package: yes
EnvManager: yes
projection: yes
history/memory: yes
external service required: yes
Skill-OPD validation: not done
```

Main gaps:

```text
Need to verify appworld_ports.ports path and server lifecycle.
Need to inspect whether execution errors/API traces are available beyond obs.
Need to decide whether collateral damage details can be exported.
Need AppWorld-specific step type taxonomy:
  API selection
  argument construction
  execution error
  verification
  irreversible operation
```

Do not run AppWorld full benchmark before Sokoban exporter validation.

### 3.3 WebShop

Current readiness:

```text
env package: yes
EnvManager: yes
projection: yes
history/memory: yes
setup/indexing required: yes
Skill-OPD validation: not done
```

Main gaps:

```text
Need to verify Google Drive data download on server.
Need to verify OpenJDK/Pyserini/Lucene indexing.
Need to inspect which search index path runtime uses.
Need to export page/action type if available.
Need WebShop-specific step type taxonomy:
  search query
  click navigation
  item comparison
  option selection
  buy decision
```

Do not run WebShop full training before AppWorld/WebShop setup audit.

## 4. What Should Not Be Added Yet

Do not add these to rollout exporter in phase 1:

```text
teacher logits
global skill
residual skill
alpha_t
distillation targets
teacher-side residual benefit score
```

Reason:

The current exporter is responsible for raw student trajectory evidence only.
Teacher scoring and Skill-OPD targets should be separate offline modules.

## 5. Next Required Server Report

After running the Sokoban server command, report:

```text
commit hash
exact command
GPU type/count
model path
number of JSONL lines
one redacted sample trajectory line or schema summary
whether required fields are present
first error trace if failed
```

Minimum acceptance response:

```text
Sokoban rollout export generated N trajectories.
Required fields present: yes/no.
Missing fields: ...
First blocker: ...
```

