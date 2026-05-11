# Skill-OPD Development Handoff

Last updated: 2026-05-11

Audience: the next Codex agent or engineer continuing `verl-agent-opd`.

Read this first, then read:

```text
tasks/phase1_sokoban_rollout.md
tasks/dataset_code_audit.md
tasks/verl_training_flow_deep_dive.md
tasks/agentic_rollout_gap_list.md
```

## 1. Current Repository State

Active repository:

```text
thinkingmanyangyang/verl-agent-opd
local path on current machine: D:\codex_workspace\skill_opd\verl-agent
main branch: main
upstream reference: langfengq/verl-agent
```

Recent Skill-OPD commits:

```text
bb7afdf Add Skill-OPD offline rollout export
b37d1c8 Add Skill-OPD task tracker
890e7c0 Audit agentic rollout data flow
3e1dcc4 Deepen Skill-OPD training flow audit
```

Current implementation stage:

```text
Stage 1 only: offline student rollout export for Sokoban.
No teacher scoring yet.
No skill retrieval yet.
No global/residual skill distillation loss yet.
No AppWorld/WebShop rollout validation yet.
```

Important boundary:

```text
This repo intentionally remains close to verl-agent.
The first implementation adds read-only rollout export and minimal metadata
plumbing. It does not rewrite the trainer, loss, advantage, or env algorithms.
```

## 2. Development Log

### 2.1 Resource and Framework Decision

Earlier exploration compared several rollout/training codebases, but the project
direction settled on:

```text
Use verl-agent as the main framework.
Do not maintain a separate external/ wrapper.
Keep upstream code mostly intact.
Add Skill-OPD-specific code under agent_system/skill_opd/.
Use minimal hooks in existing rollout/env code.
```

Reason:

```text
verl-agent already owns the hard parts:
  env reset/step loop
  vLLM rollout
  Ray trainer
  response masks
  old/ref log-prob computation
  GRPO/GiGPO advantage
  agentic env managers for Sokoban/AppWorld/WebShop
```

Skill-OPD v1 only needs reliable trajectory export before teacher scoring.

### 2.2 First Dataset Choice

First validation dataset:

```text
Sokoban
```

Why:

```text
It is fully local and does not require AppWorld services or WebShop indexes.
It has an existing verl-agent env package, prompt, projection, and trainer script.
It is enough to validate multi-turn agentic rollout export.
```

AppWorld and WebShop are not abandoned. They are documented as Phase 2/3 in:

```text
tasks/dataset_code_audit.md
tasks/dataset_reading_notes.md
```

### 2.3 Core Implementation Decision

The core hook is post-rollout, not inside the actor loss.

Chosen design:

```text
TrajectoryCollector completes env rollout
-> success_evaluator runs
-> maybe_export_rollout(...) writes JSONL
-> normal trainer continues
```

Reason:

```text
At that point, total_batch_list, total_infos, episode rewards, episode lengths,
and success metrics are all available.

This gives Skill-OPD enough data for later offline teacher scoring without
changing the RL update path.
```

Rejected for v1:

```text
Modify actor loss now.
Inject teacher logits now.
Add skill prompts into rollout now.
Change GRPO/GiGPO advantage now.
Export every intermediate trainer tensor now.
```

## 3. Core Code Map

### 3.1 New Skill-OPD Package

Directory:

```text
agent_system/skill_opd/
```

Files:

| File | Role |
|---|---|
| `README.md` | short package overview |
| `__init__.py` | package marker |
| `config.py` | parse `+skill_opd.*` Hydra overrides |
| `io.py` | JSONL writer and JSON-safe conversion helpers |
| `schema.py` | dataclasses for trajectory and step records |
| `rollout_exporter.py` | converts rollout internals into JSONL records |
| `rollout_hook.py` | small hook called from the rollout loop |

### 3.2 Export Config

File:

```text
agent_system/skill_opd/config.py
```

Purpose:

```text
Read optional Hydra config:
  +skill_opd.export_rollouts=True
  +skill_opd.export_path=...
  +skill_opd.overwrite=True
```

Behavior:

```text
If skill_opd config is missing or export_rollouts=False, exporter does nothing.
```

This keeps the feature inert unless explicitly enabled.

### 3.3 Export Schema

File:

```text
agent_system/skill_opd/schema.py
```

Main records:

```text
RolloutTrajectoryRecord
RolloutStepRecord
```

Important trajectory fields:

```text
trajectory_id
env_name
schema_version
total_reward
episode_length
success
steps
metadata
```

Important step fields:

```text
step_id
prompt_text
response_text
response_token_ids
text_action
projected_action
reward
done
active_mask
is_action_valid
info
available_keys
response_mask
rollout_log_probs
```

Why these fields:

```text
They are enough to reconstruct:
  q / h_t
  student model output
  parsed env action
  reward and done
  invalid/error/success signals

They are also enough for the next stage:
  global skill retrieval
  residual step candidate selection
  teacher-side scoring
```

### 3.4 Exporter

File:

```text
agent_system/skill_opd/rollout_exporter.py
```

Main class:

```text
RolloutExporter
```

Main method:

```text
RolloutExporter.build_records(...)
```

Input:

```text
total_batch_list
total_infos
episode_rewards
episode_lengths
success
tokenizer
```

Output:

```text
one JSON object per trajectory
```

Important behavior:

```text
prompt_text:
  decoded from per-step prompt tokens

response_text:
  decoded from response token ids or fallback text fields

response_mask:
  uses explicit response_mask if present;
  otherwise derives from attention_mask[-len(responses):]

done:
  uses exact step_data["dones"] if present;
  otherwise falls back to inferred last active step and marks metadata

text_action:
  uses step_data["text_action"], fallback info["raw_text_action"]

projected_action:
  uses step_data["projected_action"], fallback info["projected_action"]

rollout_log_probs:
  exported only if vLLM/rollout produced them
```

Important limitation:

```text
old_log_probs, ref_log_prob, advantages, and returns are not exported in v1
because they are computed later in the trainer, after this exporter hook.
```

### 3.5 Rollout Hook

File:

```text
agent_system/skill_opd/rollout_hook.py
```

Main function:

```text
maybe_export_rollout(...)
```

Behavior:

```text
Load export config.
If disabled, return immediately.
If enabled, instantiate RolloutExporter and append/write JSONL.
```

This is intentionally thin. Do not add teacher scoring here in the first
implementation.

## 4. Minimal Changes to Existing verl-agent Code

### 4.1 Rollout Loop Integration

File:

```text
agent_system/multi_turn_rollout/rollout_loop.py
```

Changes:

```text
from agent_system.skill_opd.rollout_hook import maybe_export_rollout
```

Inside the multi-turn loop, after decoding model responses and stepping env:

```text
text_actions = tokenizer.batch_decode(...)
next_obs, rewards, dones, infos = envs.step(text_actions)

batch.non_tensor_batch["text_action"] = text_actions
batch.non_tensor_batch["dones"] = dones
batch.non_tensor_batch["projected_action"] = [info["projected_action"] ...]
```

After rollout completion and success evaluation:

```text
maybe_export_rollout(...)
```

Why this minimal change is correct:

```text
The rollout loop is the only place that sees, at the same time:
  model response tokens
  decoded text actions
  env rewards/dones/infos
  total trajectory lists
```

### 4.2 EnvManager Metadata Plumbing

File:

```text
agent_system/environments/env_manager.py
```

Changes:

For relevant env managers, add to each `info` dict:

```text
info["raw_text_action"] = text_actions[i]
info["projected_action"] = actions[i]
```

Currently applied to the main action-parsing env managers, including:

```text
Sokoban
WebShop
AppWorld
AlfWorld path touched by the same pattern
```

Why this is needed:

```text
response_text is the full model output.
text_action/raw_text_action is the decoded string sent into projection.
projected_action is what the environment actually executed.

These are not equivalent, especially when output format is invalid.
```

## 5. Scripts Added

Directory:

```text
examples/skill_opd/
```

### 5.1 prepare_sokoban_data.sh

Purpose:

```text
Create small placeholder parquet files expected by verl-agent.
```

Important point:

```text
Sokoban board states are generated by env reset.
The parquet rows mainly control batch size, modality, and prompt plumbing.
```

Run:

```bash
MODE=text \
TRAIN_DATA_SIZE=8 \
VAL_DATA_SIZE=8 \
LOCAL_DIR=$HOME/data/verl-agent \
bash examples/skill_opd/prepare_sokoban_data.sh
```

### 5.2 download_qwen3_model.sh

Purpose:

```text
Download Qwen/Qwen3-4B or Qwen/Qwen3-8B with huggingface_hub.snapshot_download.
```

Run:

```bash
MODEL_NAME=Qwen/Qwen3-4B \
MODEL_DIR=$HOME/models/Qwen3-4B \
bash examples/skill_opd/download_qwen3_model.sh
```

### 5.3 run_sokoban_rollout_export_qwen3.sh

Purpose:

```text
Run validation-only Sokoban rollout and export JSONL.
```

Run:

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

Important:

```text
This requires GPU/vLLM/Ray.
Do not run it on the current CPU-only local machine.
```

Potentially confusing setting:

```text
env.sokoban.mode='tiny_rgb_array'
```

In this codebase, `tiny_rgb_array` returns a compact text board string. Only
`rgb_array` returns actual image arrays. So this setting is acceptable for the
first Qwen3 text-model smoke run.

## 6. What Has Been Verified Locally

Local CPU checks passed before this handoff:

```bash
python -m py_compile \
  agent_system/environments/env_manager.py \
  agent_system/multi_turn_rollout/rollout_loop.py \
  agent_system/skill_opd/config.py \
  agent_system/skill_opd/io.py \
  agent_system/skill_opd/schema.py \
  agent_system/skill_opd/rollout_exporter.py \
  agent_system/skill_opd/rollout_hook.py

bash -n examples/skill_opd/*.sh
git diff --check
```

Also verified:

```text
Repository was clean and synced with origin/main after commit 3e1dcc4.
```

Not verified locally:

```text
real vLLM generation
Ray GPU rollout
Sokoban JSONL from a real model
AppWorld service
WebShop data/index setup
```

## 7. Immediate Next Step for the Next Agent

Do not start with teacher scoring. First validate the exporter.

On GPU server:

```bash
cd ~/projects/verl-agent-opd
git checkout main
git pull --rebase origin main
conda activate verl-agent

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

Then inspect:

```bash
wc -l outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl
head -n 1 outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl
```

Report back into a task note:

```text
commit hash
exact command
GPU type/count
model path
JSONL path
number of trajectories
first trajectory schema summary
missing fields
first error trace if failed
```

## 8. Expected JSONL Shape

One line is one trajectory:

```json
{
  "trajectory_id": "sokoban-...",
  "env_name": "Sokoban",
  "schema_version": "skill_opd.rollout.v1",
  "total_reward": 0.0,
  "episode_length": 3,
  "success": false,
  "steps": [
    {
      "step_id": 0,
      "prompt_text": "...current Sokoban board and instructions...",
      "response_text": "<think>...</think><action>left</action>",
      "response_token_ids": [1, 2, 3],
      "text_action": "<think>...</think><action>left</action>",
      "projected_action": 3,
      "reward": -0.1,
      "done": false,
      "active_mask": true,
      "is_action_valid": true,
      "info": {
        "won": false,
        "action_is_effective": true,
        "raw_text_action": "...",
        "projected_action": 3
      },
      "available_keys": ["..."],
      "response_mask": [1, 1, 1],
      "rollout_log_probs": null
    }
  ],
  "metadata": {}
}
```

The exact field values may differ, but these fields are the target.

## 9. Known Risks and Likely Failure Points

### 9.1 vLLM or Ray Runtime Failure

Likely cause:

```text
server environment mismatch, CUDA/vLLM version, model memory, Ray resource config.
```

Action:

```text
Do not rewrite Skill-OPD code first.
Check base verl-agent Sokoban script with the same environment.
Then compare with examples/skill_opd/run_sokoban_rollout_export_qwen3.sh.
```

### 9.2 Missing rollout_log_probs

This is acceptable for v1.

Reason:

```text
Trainer recomputes old_log_probs later.
rollout_log_probs is diagnostic at this stage.
```

Action:

```text
Record as missing optional field, not blocker.
```

### 9.3 done Field Missing or Misaligned

The exporter has fallback inference, but real rollout should use exact `dones`.

Action:

```text
If JSONL marks metadata.done_is_inferred_from_episode_length=true unexpectedly,
inspect rollout_loop.py around batch.non_tensor_batch["dones"].
```

### 9.4 prompt_text Does Not Fully Represent h_t

If exported prompt lacks history/current board:

```text
inspect SokobanEnvironmentManager.build_text_obs
inspect TrajectoryCollector.preprocess_batch
consider exporting raw memory_context in v1.1
```

Do not add memory export until a real JSONL proves prompt_text is insufficient.

### 9.5 Duplicate or Empty Export Lines

Likely cause:

```text
hook called in both train and validation paths
overwrite/append behavior
Ray rank duplication
```

Action:

```text
First check trainer.val_only=True and export path.
Then inspect maybe_export_rollout call count.
```

## 10. What Not to Do Next

Do not immediately implement:

```text
global skill retrieval
residual skill generation
teacher logits
distillation target tensors
actor loss modification
AppWorld/WebShop training runs
multi-skill mixture
large benchmark runs
```

Reason:

```text
The first dependency is validated student trajectory JSONL.
Without real rollout data, teacher scoring design cannot be grounded.
```

## 11. After Sokoban JSONL Works

Next development sequence:

```text
1. Add a lightweight JSONL inspection script if needed.
2. Confirm prompt_text is sufficient h_t.
3. Define offline teacher-scoring input schema from exported trajectories.
4. Implement global skill bank placeholder and retrieval interface.
5. Implement candidate step selection from JSONL:
   invalid action
   high response uncertainty if available
   final/success/failure step
   repeated ineffective action
6. Only then design teacher scoring calls.
```

Keep teacher scoring offline at first:

```text
student rollout JSONL
-> global skill retrieval
-> candidate step selection
-> residual skill retrieval/summarization
-> teacher global/residual scoring
-> m_t and alpha_t*
-> separate teacher-score JSONL/parquet
```

Do not mix this with RL training until the offline artifacts are validated.

