# verl-agent Training Flow Deep Dive for Skill-OPD

Last updated: 2026-05-07

This note traces the training data path that matters for Skill-OPD:

```text
parquet row
-> RLHFDataset item
-> collate_fn batch
-> DataProto
-> RayPPOTrainer.fit
-> TrajectoryCollector multi-turn rollout
-> EnvManager reset/step
-> rollout batch
-> reward/log-prob/advantage
-> actor loss
```

The conclusion is that Skill-OPD v1 should not modify the trainer loss path yet.
The safe first hook is the post-rollout trajectory exporter. Teacher scoring and
distillation can be added later after the exported trajectory schema is validated
on real Sokoban rollouts.

## 1. Entry Point and Boot Sequence

Primary entry:

```text
verl/trainer/main_ppo.py::main
verl/trainer/main_ppo.py::run_ppo
verl/trainer/main_ppo.py::TaskRunner.run
```

`TaskRunner.run` is the top-level runtime assembly point.

Important operations:

| Step | Code path | Output |
|---|---|---|
| Resolve config | `OmegaConf.resolve(config)` | fully materialized Hydra config |
| Copy model path | `copy_to_local(config.actor_rollout_ref.model.path)` | local HF model path |
| Build envs | `agent_system.environments.make_envs(config)` | train env manager and val env manager |
| Build tokenizer | `hf_tokenizer(local_path, trust_remote_code=...)` | tokenizer used by dataset and rollout |
| Select workers | FSDP or Megatron branch | actor, rollout, critic, reference worker classes |
| Build reward manager | `EpisodeRewardManager` when `reward_model.reward_manager=episode` | episode-level reward tensor writer |
| Build collector | `TrajectoryCollector(config, tokenizer, processor)` | agentic rollout orchestrator |
| Build datasets | `create_rl_dataset(...)` | train and validation datasets |
| Build trainer | `RayPPOTrainer(...)` | full PPO/GRPO/GiGPO training controller |

Important env-specific detail:

```text
assert config.actor_rollout_ref.rollout.n == 1
```

For env-based agentic training, rollout multiplicity is handled by
`config.env.rollout.n`, not by the standard verl rollout `n`. This matters for
GRPO/GiGPO grouping because repeated trajectories come from env grouping.

## 2. Dataset -> DataProto

Relevant files:

```text
examples/data_preprocess/prepare.py
verl/utils/dataset/rl_dataset.py
verl/protocol.py
verl/trainer/ppo/ray_trainer.py
```

### 2.1 Parquet Row

The standard row expected by `RLHFDataset` contains:

| Field | Type | Purpose |
|---|---|---|
| `prompt` | chat message list or string-like prompt field | source prompt before chat template |
| `data_source` | string | dataset/env identifier |
| `extra_info` | dict | index, split metadata, optional env kwargs |
| `reward_model` | optional dict | non-env reward metadata, usually unused for episode env reward |
| multimodal fields | optional | images/videos if processor is enabled |

For agentic env tasks, the parquet row is usually lightweight. The actual task
state is controlled by the EnvManager and env reset, not by a large static
supervised target.

### 2.2 RLHFDataset Item

`verl/utils/dataset/rl_dataset.py::RLHFDataset.__getitem__` performs:

```text
row["prompt"]
-> tokenizer.apply_chat_template(...)
-> tokenization
-> pad/truncate
-> input_ids / attention_mask / position_ids
-> raw_prompt_ids
-> preserve non-token metadata
```

Important fields emitted:

| Field | Where emitted | Meaning |
|---|---|---|
| `input_ids` | dataset item tensor | padded prompt tokens |
| `attention_mask` | dataset item tensor | prompt attention mask |
| `position_ids` | dataset item tensor | prompt position ids |
| `raw_prompt_ids` | dataset item non-tensor/list | unpadded prompt token ids |
| `raw_prompt` | optional | raw chat messages if configured |
| `full_prompts` | optional | full prompt string if configured |
| `index` | from `extra_info.index` | sample index |
| `tools_kwargs` | from `extra_info.tools_kwargs` or `{}` | tool metadata if present |
| `data_source` | copied from row | task/source name |
| `extra_info` | copied from row | additional metadata |

### 2.3 collate_fn

`verl/utils/dataset/rl_dataset.py::collate_fn` converts a list of dataset items:

```text
torch.Tensor fields -> torch.stack
non-tensor fields -> np.ndarray(dtype=object)
```

This is important because `DataProto.from_single_dict` requires non-tensor
fields to be numpy arrays. Arbitrary Python lists cannot be put directly into
`DataProto.non_tensor_batch`.

### 2.4 DataProto

`verl/protocol.py::DataProto` has three containers:

| Container | Content |
|---|---|
| `batch` | tensor fields in a TensorDict |
| `non_tensor_batch` | numpy object arrays for metadata |
| `meta_info` | scalar or global run metadata |

Key methods:

| Method | Role |
|---|---|
| `from_single_dict` | splits tensor and numpy fields into `batch` and `non_tensor_batch` |
| `pop` | removes selected fields from the current DataProto and returns a new DataProto |
| `union` | merges another DataProto into the current one |
| `repeat` | repeats tensors and non-tensors along the batch dimension |

Critical behavior:

```text
batch.pop(...) is destructive.
```

If `RayPPOTrainer.fit` pops fields into `gen_batch`, those fields are removed
from the original `batch` unless they are later unioned back.

### 2.5 Trainer Dataloader

`verl/trainer/ppo/ray_trainer.py::_create_dataloader` creates:

```text
train_dataset
-> DataLoader(collate_fn=collate_fn)
-> batch_dict
-> DataProto.from_single_dict(batch_dict)
```

At this point the data is still a prompt batch. No env step has happened.

## 3. DataProto -> Rollout

Relevant files:

```text
verl/trainer/ppo/ray_trainer.py
agent_system/multi_turn_rollout/rollout_loop.py
verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
agent_system/environments/env_manager.py
```

### 3.1 gen_batch Creation

Inside `RayPPOTrainer.fit`, the trainer separates generation inputs from the
main batch:

```text
gen_batch = batch.pop(batch_keys=[...prompt tensor keys...],
                      non_tensor_batch_keys=[...prompt metadata keys...])
```

The exact key set is config-dependent, but the intent is stable:

```text
gen_batch contains prompt tensors and prompt metadata.
batch is later replaced by rollout output for agentic env training.
```

### 3.2 Agentic Rollout Hook

Instead of calling plain vLLM once, env training calls:

```text
self.traj_collector.multi_turn_loop(
    gen_batch,
    actor_rollout_wg,
    envs,
    is_train=True,
)
```

The collector owns:

```text
reset envs
build step prompt
call model
decode response
project response into env action
env.step
store memory
repeat until done/max_steps
gather rollout data
```

### 3.3 h_t Construction

For Skill-OPD, the practical `h_t` is the full prompt text sent to the model at
step `t`.

In code, it is assembled by:

```text
EnvManager.reset/step
-> prompt builder in agent_system/environments/prompts/*.py
-> memory state from agent_system/memory/memory.py
-> TrajectoryCollector.preprocess_batch
-> tokenized batch input to vLLM
```

The exporter should preserve `prompt_text` because it is the most reliable
reconstruction of:

```text
task instruction + current observation + visible interaction history
```

If future work needs a cleaner state representation, it can additionally export
`memory_context` or raw env observation. That should be v1.1, not v1.

### 3.4 vLLM Output Fields

`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py::vLLMRollout.generate_sequences`
returns a `DataProto` with generation tensors.

Important fields expected downstream:

| Field | Meaning |
|---|---|
| `prompts` | prompt token ids |
| `responses` | generated response token ids |
| `input_ids` | prompt + response token ids |
| `attention_mask` | prompt + response attention mask |
| `position_ids` | prompt + response position ids |
| `rollout_log_probs` | rollout-time token log probabilities, if configured/available |

`rollout_log_probs` is useful for diagnostics and possibly future OPD logging,
but the trainer recomputes `old_log_probs` before actor update. It should not be
treated as the authoritative training log prob unless the trainer explicitly does
so.

### 3.5 Response Decoding and Projection

`TrajectoryCollector` decodes model responses into text actions:

```text
responses
-> tokenizer.decode(...)
-> text_actions
-> EnvManager.step(text_actions)
-> dataset-specific projection
-> projected_action
-> env.step(projected_action)
```

For Skill-OPD we must distinguish:

| Field | Meaning |
|---|---|
| `response_text` | full decoded model output |
| `text_action` | decoded text sent into projection |
| `projected_action` | parsed action actually sent to environment |
| `is_action_valid` | projection validity, not necessarily env effectiveness |

This distinction is necessary because residual skills should be analyzed against
the model decision and the environment action separately.

## 4. Rollout -> Training Batch

Relevant files:

```text
agent_system/multi_turn_rollout/rollout_loop.py
agent_system/multi_turn_rollout/utils.py
agent_system/skill_opd/rollout_hook.py
agent_system/skill_opd/rollout_exporter.py
```

### 4.1 Per-Step Collection

During multi-turn rollout, the collector appends per-step `DataProto` objects to
`total_batch_list` and per-step env infos to `total_infos`.

Important per-step fields:

| Field | Source | Purpose |
|---|---|---|
| `prompts` | vLLM output | prompt token ids for this step |
| `responses` | vLLM output | model response token ids |
| `input_ids` | vLLM output | prompt + response |
| `attention_mask` | vLLM output | token mask |
| `position_ids` | vLLM output | position ids |
| `rollout_log_probs` | vLLM output if enabled | rollout-time log probs |
| `is_action_valid` | EnvManager info | projection validity |
| `text_action` | decoded model text | raw text action sent to projection |
| `projected_action` | EnvManager info | action sent to env |
| `dones` | env step output | exact env done flag |
| `step_rewards` | env reward path | step-level reward where configured |

### 4.2 active_masks and Finished Episodes

`active_masks` tracks which env instances are still active. It prevents finished
episodes from continuing to contribute rollout/training samples.

The practical invariant is:

```text
If active_masks[i] is false at a later step, sample i should not produce a new
training step for that later step.
```

For exporter correctness, exact `dones` are better than inferring final step
from active mask. The current Skill-OPD exporter supports exact `dones` when the
rollout loop provides them, and falls back to inferred final-step metadata for
older records.

### 4.3 gather_rollout_data

`TrajectoryCollector.gather_rollout_data` flattens multi-step rollout data into
a batch suitable for the trainer.

Important consequence:

```text
Training sees step-level samples, not one nested trajectory object.
```

Therefore the exporter should run before or during this transition while
`total_batch_list`, `total_infos`, rewards, lengths, and success are still
available. This is why the current Skill-OPD hook is placed after rollout
completion, not inside actor loss.

### 4.4 Exporter Position

Current Skill-OPD hook:

```text
TrajectoryCollector
-> rollout complete
-> success_evaluator
-> maybe_export_rollout(...)
-> return rollout batch to trainer
```

This is the correct v1 placement because:

```text
it is read-only with respect to training
it has full trajectory context
it does not interfere with PPO/GRPO/GiGPO loss
it can be disabled through config
```

## 5. Reward / Log Prob / Advantage

Relevant files:

```text
verl/trainer/ppo/ray_trainer.py
verl/trainer/ppo/reward.py
agent_system/reward_manager/episode.py
verl/trainer/ppo/core_algos.py
gigpo/core_gigpo.py
```

### 5.1 response_mask

`verl/trainer/ppo/ray_trainer.py::compute_response_mask` returns:

```text
attention_mask[:, -response_length:]
```

where `response_length = batch.batch["responses"].size(1)`.

Meaning:

```text
Only response tokens are eligible for loss/reward masking.
Prompt tokens and env observation tokens inside the prompt are not directly
optimized as response tokens.
```

If `multi_turn=True`, downstream advantage/loss logic may use a stricter
`loss_mask`. This should be audited on the server config used for real runs.

### 5.2 Episode Reward Tensor

`agent_system/reward_manager/episode.py::EpisodeRewardManager.__call__` converts
episode-level env reward into token-level reward tensor.

For each step sample:

```text
valid_response_length = attention_mask[prompt_length:].sum()
reward_tensor[i, valid_response_length - 1] = episode_reward_or_normalized_reward
```

Thus env reward is assigned to the final valid response token for that step
sample. This is standard for response-level RL with token-level tensors.

### 5.3 Invalid Action Penalty

`ray_trainer.py::apply_invalid_action_penalty` reads:

```text
data_item.non_tensor_batch["is_action_valid"]
```

If invalid:

```text
subtract coefficient from the last valid response token reward
subtract coefficient from step_rewards if step_rewards exists
```

This means invalid action penalty affects RL reward, but it does not directly
change the prompt, projection, or env state.

### 5.4 old_log_probs

`RayPPOTrainer.fit` recomputes old policy log probs before actor update:

```text
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)
```

This is the training-time policy log prob used by PPO/GRPO-style losses.

Important distinction:

```text
rollout_log_probs: produced during generation, used for diagnostics if present
old_log_probs: recomputed before update, used by actor loss
```

### 5.5 ref_log_prob

If reference policy KL is enabled, the trainer computes:

```text
ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
batch = batch.union(ref_log_prob)
```

If reference policy is not enabled, `ref_log_prob` is not available and
`apply_kl_penalty` is skipped unless the config demands it.

### 5.6 token_level_scores and token_level_rewards

The trainer first obtains:

```text
token_level_scores = reward_tensor
```

Then:

```text
if algorithm.use_kl_in_reward:
    token_level_rewards = token_level_scores - kl_penalty
else:
    token_level_rewards = token_level_scores
```

Skill-OPD distillation should not be added by modifying `token_level_rewards`
in v1. That would mix reward shaping and teacher supervision too early.

### 5.7 Advantage

`ray_trainer.py::compute_advantage` dispatches by estimator.

Important fields:

| Estimator | Key fields |
|---|---|
| GRPO | `token_level_rewards`, `response_mask`, `uid`, `traj_uid` |
| GiGPO | `token_level_rewards`, `step_rewards`, `anchor_obs`, `uid`, `traj_uid` |

`uid` and `traj_uid` are grouping keys. `anchor_obs` is especially relevant for
GiGPO step grouping and must be preserved by env rollout if GiGPO is used.

## 6. Actor Loss

Relevant file:

```text
verl/trainer/ppo/core_algos.py
```

The actor update expects a batch containing at least:

```text
responses
attention_mask
response_mask or loss_mask
old_log_probs
advantages
token_level_rewards
```

Optional but common:

```text
ref_log_prob
values
returns
rollout_log_probs
```

The important Skill-OPD conclusion:

```text
The RL actor loss consumes flattened step-level response samples.
It does not naturally consume nested trajectories or future teacher targets yet.
```

Therefore, teacher scoring should first consume exported JSONL offline. Only
after the teacher target schema is validated should we add an auxiliary
distillation loss into actor update.

## 7. Where Skill-OPD Can Be Added Later

Recommended staged integration:

| Stage | Add Skill-OPD here? | Reason |
|---|---|---|
| Env reset/step | No for v1 | Keep environment behavior identical to upstream |
| Rollout exporter | Yes now | Safe read-only trajectory capture |
| Offline teacher scorer | Yes next | Computes global/residual teacher shift without training changes |
| Dataset/DataProto | Later | Needed after teacher targets exist |
| Actor loss | Later | Requires validated target tensors and masks |
| Reward/advantage | No initially | Distillation is supervision, not env reward |

Future distillation hook candidates:

```text
Option A: add teacher target tensors to DataProto after rollout and before actor update.
Option B: create an offline distillation dataset from exported JSONL + teacher scores.
Option C: add an auxiliary loss inside actor worker update once target masking is stable.
```

For the current project phase, Option B is the lowest-risk path.

## 8. Field Flow Table

| Field | First known source | Modified by | Used by training | Export readiness |
|---|---|---|---|---|
| `input_ids` | dataset and vLLM rollout | rollout appends response | yes | useful for debug |
| `attention_mask` | dataset and vLLM rollout | rollout appends response mask | yes | used to derive response mask |
| `position_ids` | dataset and vLLM rollout | rollout updates | yes | optional export |
| `raw_prompt_ids` | `RLHFDataset.__getitem__` | usually metadata only | generation metadata | optional |
| `raw_prompt` | dataset if configured | not central | no | optional |
| `data_source` | parquet row | carried in metadata | grouping/logging | optional |
| `env_kwargs` | config/extra_info/env manager | env reset | env behavior | optional |
| `prompts` | rollout | per-step prompt tokens | yes | export prompt_text after decode |
| `responses` | rollout | generated tokens | yes | export response ids/text |
| `rollout_log_probs` | vLLM rollout | diagnostics | diagnostic only | export if present |
| `response_mask` | `compute_response_mask` | maybe loss mask branch | yes | exporter can derive |
| `text_action` | decoded model response | EnvManager receives it | no direct loss | export required |
| `projected_action` | dataset projection | EnvManager info | no direct loss | export required |
| `rewards` | env.step | reward manager converts | yes | export required |
| `dones` | env.step | active mask update | rollout control | export required |
| `active_masks` | rollout loop | updated each step | filters rollout | export metadata |
| `is_action_valid` | projection | invalid penalty | yes if penalty enabled | export required |
| `uid` | rollout/grouping | trainer | GRPO/GiGPO | export optional |
| `traj_uid` | rollout/grouping | trainer | GRPO/GiGPO | export optional |
| `episode_rewards` | env rollout summary | reward manager | yes | trajectory-level export |
| `episode_lengths` | env rollout summary | reward manager | yes | trajectory-level export |
| `success` | EnvManager success evaluator | exporter/logging | metrics | trajectory-level export |
| `total_infos` | EnvManager step infos | exporter | not actor loss | export info |
| `token_level_scores` | reward manager | KL penalty path | yes | not v1 export |
| `token_level_rewards` | reward + KL | advantage | yes | not v1 export |
| `old_log_probs` | actor worker recompute | actor loss | yes | not available at exporter hook |
| `ref_log_prob` | reference worker | KL | yes if enabled | not available at exporter hook |
| `advantages` | trainer algorithms | actor loss | yes | not v1 export |
| `returns` | advantage/value path | critic/loss | yes if critic | not v1 export |

## 9. Hypothetical Sokoban Step Trace

```text
parquet row:
  prompt: generic Sokoban task prompt
  extra_info.index: 0

RLHFDataset item:
  input_ids: prompt tokens
  attention_mask: prompt mask
  raw_prompt_ids: unpadded prompt ids

RayPPOTrainer.fit:
  gen_batch = prompt tensors + metadata

TrajectoryCollector step 0:
  env.reset(seed)
  obs text: board state
  prompt_text: instruction + board + action format
  vLLM response: "<think>...</think><action>left</action>"
  text_action: same decoded response
  sokoban_projection: action id for left, valid=true
  env.step(left)
  reward: step reward
  done: false
  info: won=false, action_is_effective=true, projected_action="left"
  memory.store(previous_obs, "left")

Exporter:
  trajectory_id: sokoban-...
  step_id: 0
  prompt_text: full h_t
  response_text: full decoded response
  response_token_ids: generated token ids
  text_action: decoded model response
  projected_action: left
  reward: step reward
  done: false
  is_action_valid: true
  info: env info
```

## 10. Validation Checklist

Local CPU checks:

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
```

Server-only checks:

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

Server report must include:

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

