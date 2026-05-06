# Skill-OPD Dataset and Agentic Rollout Code Audit

Last updated: 2026-05-07

This document audits how `Sokoban`, `AppWorld`, and `WebShop` flow through
`verl-agent-opd`: data preparation, dataset loading, agentic environment
interaction, rollout collection, reward/advantage computation, and the
Skill-OPD offline rollout exporter.

The immediate execution target is still Sokoban. AppWorld and WebShop are
audited here so that later integration work has a concrete code map instead of
starting from scratch.

## 1. Current Status

| Dataset | Current status | Skill-OPD status | Recommended phase |
|---|---|---|---|
| Sokoban | Environment, prompt, projection, EnvManager, rollout scripts exist. | Offline JSONL exporter is implemented and now includes raw text action, projected action, exact `done` when available, response mask, and rollout log probs when present. Needs real GPU validation. | Phase 1 |
| AppWorld | Environment package and EnvManager branch exist. Requires external AppWorld service instances. | Not validated for Skill-OPD export. Needs server/setup audit and AppWorld-specific schema. | Phase 2 |
| WebShop | Environment package, setup scripts, indexing scripts, and EnvManager branch exist. Requires product data and search index. | Not validated for Skill-OPD export. Needs setup/index check and WebShop-specific schema. | Phase 3 |

Important distinction:

```text
verl-agent env package exists != Skill-OPD pipeline is complete
```

For Skill-OPD, a dataset is complete only when we can export enough trajectory
state to reconstruct:

```text
q, h_t, raw model response, projected env action, reward, done, info, success
```

## 2. End-to-End Training Data Flow

The main training/rollout path is:

```text
examples/data_preprocess/prepare.py or env-specific setup
-> parquet train_files / val_files
-> verl/utils/dataset/rl_dataset.py::RLHFDataset
-> torch DataLoader
-> verl.DataProto
-> verl/trainer/ppo/ray_trainer.py::RayPPOTrainer.fit
-> gen_batch
-> agent_system/multi_turn_rollout/rollout_loop.py::TrajectoryCollector
-> EnvManager.reset()
-> preprocess_batch()
-> vLLM generate_sequences()
-> tokenizer.decode(responses)
-> EnvManager.step(text_actions)
-> total_batch_list / total_infos
-> gather_rollout_data()
-> compute response_mask / rewards / old_log_probs / ref_log_probs
-> compute advantage
-> actor update
```

### 2.1 Parquet Row

Source: `examples/data_preprocess/prepare.py`

The current generic preparation script uses `hiyouga/geometry3k` only as a
placeholder data source to create rows with the right modality and batch size.
For Sokoban and WebShop, the real task state comes from the environment, not
from the parquet row.

Typical text-mode row:

```text
data_source: "text"
prompt: [{"role": "user", "content": ""}]
ability: "agent"
extra_info:
  split: train/test
  index: int
```

Typical visual-mode row:

```text
data_source: "visual"
prompt: [{"role": "user", "content": "<image>"}]
images: image payload from placeholder dataset
ability: "agent"
extra_info:
  split: train/test
  index: int
```

Observation:

- Sokoban/WebShop do not rely on this row for task content.
- AppWorld task content is selected by its env wrapper via AppWorld task ids.
- The parquet controls batch size, modality, initial prompt shape, and prompt
  index/grouping metadata.

### 2.2 RLHFDataset Item

Source: `verl/utils/dataset/rl_dataset.py`

Relevant methods:

- `RLHFDataset._read_files_and_tokenize`
- `RLHFDataset.__getitem__`
- `collate_fn`

`__getitem__` builds:

```text
input_ids
attention_mask
position_ids
raw_prompt_ids
raw_prompt, if data.return_raw_chat=True
full_prompts, if data.return_full_prompt=True
multi_modal_data / multi_modal_inputs, if processor is used
index
tools_kwargs
data_source
extra_info-derived fields
```

`collate_fn` stacks tensor fields and converts non-tensor fields to object
numpy arrays. This is why rollout metadata such as `raw_prompt_ids`, `uid`, and
`traj_uid` are carried in `DataProto.non_tensor_batch`.

### 2.3 Trainer Batch to Rollout Batch

Source: `verl/trainer/ppo/ray_trainer.py`

In `RayPPOTrainer.fit`, each DataLoader batch becomes a `DataProto`. Prompt
fields are popped to create `gen_batch`, which is passed to the rollout worker:

```text
batch tensors:
  input_ids
  attention_mask
  position_ids

non_tensor_batch:
  raw_prompt_ids
  data_source
  raw_prompt?
  multi_modal_data?
  tools_kwargs?
  env_kwargs?
```

For multi-turn agentic rollout, the original static prompt is not the actual
step prompt after reset. `TrajectoryCollector.preprocess_batch()` rebuilds the
prompt from current environment observation at every step.

### 2.4 Agentic Rollout Loop

Source: `agent_system/multi_turn_rollout/rollout_loop.py`

The central method is `TrajectoryCollector.vanilla_multi_turn_loop`.

Flow:

```text
obs, infos = envs.reset(...)
for step in range(env.max_steps):
  active_masks = not is_done
  batch = preprocess_batch(gen_batch, obs)
  batch_input = batch.pop(input_ids, attention_mask, position_ids, raw_prompt_ids, ...)
  batch_output = actor_rollout_wg.generate_sequences(batch_input)
  batch = batch.union(batch_output)
  text_actions = tokenizer.batch_decode(batch.batch["responses"])
  next_obs, rewards, dones, infos = envs.step(text_actions)
  batch.non_tensor_batch["is_action_valid"] = ...
  batch.non_tensor_batch["text_action"] = text_actions
  batch.non_tensor_batch["projected_action"] = info["projected_action"], if present
  batch.non_tensor_batch["dones"] = dones
  batch.non_tensor_batch["rewards"] = rewards
  batch.non_tensor_batch["active_masks"] = active_masks
  total_batch_list[i].append(to_list_of_dict(batch)[i])
  total_infos[i].append(infos[i])
  is_done = is_done or dones
```

The Skill-OPD hook runs after `success_evaluator`:

```text
maybe_export_rollout(...)
```

This is intentionally after full rollout collection, not inside each step.

### 2.5 vLLM Rollout Output

Source: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

`generate_sequences()` returns:

```text
prompts
responses
input_ids
rollout_log_probs
attention_mask
position_ids
```

`rollout_log_probs` are generated by vLLM when logprobs are available. The
trainer later recomputes actor `old_log_probs` and compares them with rollout
log probs for diagnostics.

### 2.6 Flattened Training Batch

Source: `TrajectoryCollector.gather_rollout_data`

Only active steps are flattened into the training batch:

```text
for data in total_batch_list[trajectory]:
  if data["active_masks"]:
    data["episode_rewards"] = episode_rewards[trajectory]
    data["episode_lengths"] = episode_lengths[trajectory]
    data["tool_callings"] = tool_callings[trajectory]
    data[success_metric] = mean success metric
    effective_batch.append(data)
```

Then:

```text
DataProto.from_single_dict(collate_fn(effective_batch))
```

This means each active environment step becomes one training sample.

### 2.7 Response Mask, Reward, Log Prob, Advantage, Loss

Source: `verl/trainer/ppo/ray_trainer.py`

Response mask:

```text
compute_response_mask(data):
  responses = data.batch["responses"]
  response_length = responses.size(1)
  attention_mask = data.batch["attention_mask"]
  return attention_mask[:, -response_length:]
```

Therefore only response tokens are used for reward/loss calculations. Prompt
and environment observation tokens condition the model, but they are not the
actor loss target.

Old log probs:

```text
old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
batch = batch.union(old_log_prob)
```

Reference log probs:

```text
ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
batch = batch.union(ref_log_prob)
```

Rewards:

```text
reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
batch.batch["token_level_scores"] = reward_tensor
```

Invalid action penalty:

```text
if actor.use_invalid_action_penalty:
  token_level_scores[last_valid_response_token] -= invalid_action_penalty_coef
```

KL-in-reward, if enabled:

```text
token_level_rewards = token_level_scores - beta * KL(old_log_probs, ref_log_prob)
```

Otherwise:

```text
token_level_rewards = token_level_scores
```

Advantage:

- GRPO groups by `uid` and `traj_uid`.
- GiGPO uses `step_rewards`, `anchor_obs`, `uid`, and `traj_uid`.
- PPO/GAE uses critic values if configured.

Actor loss:

```text
compute_policy_loss(old_log_prob, log_prob, advantages, response_mask)
```

## 3. Sokoban Audit

### 3.1 Relevant Files

```text
examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
examples/gigpo_trainer/run_sokoban.sh
examples/grpo_trainer/run_sokoban.sh
agent_system/environments/env_package/sokoban/envs.py
agent_system/environments/env_package/sokoban/projection.py
agent_system/environments/prompts/sokoban.py
agent_system/environments/env_manager.py
agent_system/multi_turn_rollout/rollout_loop.py
agent_system/skill_opd/rollout_exporter.py
```

### 3.2 Environment Construction

Source: `agent_system/environments/env_manager.py::make_envs`

When `env.env_name` contains `sokoban`, the manager builds:

```text
build_sokoban_envs(
  seed=config.env.seed,
  env_num=config.data.train_batch_size,
  group_n=config.env.rollout.n,
  mode=config.env.sokoban.mode,
  is_train=True,
  env_kwargs={
    dim_room,
    num_boxes,
    max_steps,
    search_depth
  }
)
```

Validation envs use `seed + 1000` and `group_n=1`.

### 3.3 Reset

Source: `SokobanMultiProcessEnv.reset`

Train seeds are sampled from `[0, 2**16 - 1)`. Validation seeds are sampled from
`[2**16, 2**32 - 1)`. Each sampled seed is repeated by `group_n` so grouped
rollouts start from the same room state.

Reset returns:

```text
obs_list
info_list
```

`SokobanEnvironmentManager.reset` converts this to:

Text mode:

```text
observations = {
  "text": build_text_obs(infos, obs, init=True),
  "image": None,
  "anchor": obs
}
```

Visual mode:

```text
observations = {
  "text": visual prompt,
  "image": rgb obs,
  "anchor": rgb obs
}
```

### 3.4 Prompt and History

Source:

- `agent_system/environments/prompts/sokoban.py`
- `SokobanEnvironmentManager.build_text_obs`

Initial prompt uses `SOKOBAN_TEMPLATE_NO_HIS`. Later steps use
`SOKOBAN_TEMPLATE` with:

```text
step_count
history_length
action_history
current_step
current_observation
```

History is maintained by `SimpleMemory`. After each env step:

```text
self.memory.store({
  "text_obs": self.pre_text_obs,
  "action": [ACTION_LOOKUP[act] for act in actions]
})
```

This means `h_t` is effectively the current rendered observation plus bounded
history injected into the prompt.

### 3.5 Action Projection

Source: `agent_system/environments/env_package/sokoban/projection.py`

The model is expected to emit a response containing:

```text
<think>...</think><action>...</action>
```

The projection function extracts the action span and maps it to Sokoban action
ids. The EnvManager now stores:

```text
info["raw_text_action"] = original decoded model response
info["projected_action"] = parsed env action
info["is_action_valid"] = parsed validity flag
```

`rollout_loop.py` also stores:

```text
batch.non_tensor_batch["text_action"]
batch.non_tensor_batch["projected_action"]
batch.non_tensor_batch["dones"]
```

### 3.6 Step, Reward, Done, Info

Source: `SokobanWorker.step`

Each worker returns:

```text
obs
reward
done
info
```

The manager adds:

```text
is_action_valid
raw_text_action
projected_action
```

The rollout loop adds:

```text
rewards
dones
active_masks
uid
traj_uid
```

### 3.7 Skill-OPD Export

Source: `agent_system/skill_opd/rollout_exporter.py`

Current exported per-trajectory fields:

```text
trajectory_id
env_name
total_reward
episode_length
success
tool_call_count
steps
success_metrics
metadata
```

Current exported per-step fields:

```text
step_id
prompt_text
response_text
response_token_ids
response_mask
rollout_log_probs
reward
done
active_mask
is_action_valid
text_action
projected_action
uid
traj_uid
data_source
info
available_keys
```

`done` now uses real `dones` from the rollout loop when present. If an older
rollout does not contain `dones`, exporter falls back to last active step and
sets:

```text
metadata.done_is_inferred_from_episode_length = true
```

### 3.8 Sokoban Validation Requirement

Server command:

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

Acceptance:

```text
JSONL has at least 8 trajectories.
Every trajectory has at least one step.
Each step has prompt_text, response_text, response_token_ids, text_action,
projected_action, reward, done, is_action_valid, info.
response_mask is present if attention_mask/responses are present.
rollout_log_probs is present if vLLM returns log probs.
```

## 4. AppWorld Audit

### 4.1 Relevant Files

```text
examples/env_server/start_appworld_server.sh
agent_system/environments/env_package/appworld/envs.py
agent_system/environments/env_package/appworld/projection.py
agent_system/environments/prompts/appworld.py
agent_system/environments/env_manager.py
```

### 4.2 External Service

`examples/env_server/start_appworld_server.sh` starts one `appworld serve
environment` process per required environment instance.

It computes:

```text
total_instances = train_batch_size * group_size + val_batch_size
```

It writes selected ports to:

```text
appworld_ports.ports
```

`AppWorldEnvs` then reads this file through `load_available_ports`.

### 4.3 Environment Construction

Source: `make_envs`

```text
build_appworld_envs(
  dataset_name="train",
  seed=config.env.seed,
  env_num=config.data.train_batch_size,
  group_n=group_n,
  start_server_id=0
)

build_appworld_envs(
  dataset_name="test_normal",
  seed=config.env.seed + 1000,
  env_num=config.data.val_batch_size,
  group_n=1,
  start_server_id=config.data.train_batch_size * group_n
)
```

### 4.4 Reset

Source: `AppWorldEnvs.reset`

Reset samples task ids from:

```text
load_task_ids(dataset_name)
```

Then repeats by `group_n`.

`AppWorldWorker.reset(task_id)` creates:

```text
AppWorld(
  task_id=task_id,
  experiment_name=f"default_{worker_id}",
  remote_environment_url=f"http://0.0.0.0:{port}"
)
```

Reset output:

```text
obs = self.env.task.instruction
info = {
  "task_id": task_id,
  "supervisor": dict(self.env.task.supervisor)
}
```

`AppWorldEnvironmentManager.reset` stores:

```text
self.supervisors
self.tasks
self.pre_text_obs
```

and builds the initial prompt with supervisor fields and task description.

### 4.5 Action Projection and Step

Source: `appworld_projection`

Expected model response:

```text
<think>...</think><code>python code</code>
```

Projection extracts code between `<code>` tags. If missing, it marks the action
invalid and keeps the last 100 characters as a fallback action string.

`AppWorldWorker.step(action)` executes:

```text
obs = self.env.execute(action)
done = self.env.task_completed() or current_step_count >= max_interactions
```

If done:

```text
is_success = self.env.evaluate().success
reward = 10.0 if success else 0.0
info = {"won": is_success, "step_count": current_step_count}
```

Otherwise:

```text
reward = 0.0
info = {"won": False, "step_count": current_step_count}
```

The manager now adds:

```text
is_action_valid
raw_text_action
projected_action
```

### 4.6 Prompt and History

Source: `AppWorldEnvironmentManager.build_text_obs`

Initial prompt uses task description and supervisor identity fields. Later
prompts include bounded history:

```text
Code k:
<previous action>

Result k:
<previous environment observation>
```

History is truncated to the last 10,000 characters if too long.

### 4.7 AppWorld Gaps

Not yet validated:

```text
appworld_ports.ports location relative to training command
server process lifecycle
task_id split availability on server
whether execute errors are fully present in obs/info
whether collateral damage details can be exported
whether API trace is accessible beyond plain obs
```

Skill-OPD likely needs AppWorld-specific fields:

```text
task_id
supervisor
code_action
execution_result
step_count
task_completed
success
evaluate result details, if available
```

## 5. WebShop Audit

### 5.1 Relevant Files

```text
examples/gigpo_trainer/run_webshop.sh
examples/grpo_trainer/run_webshop.sh
examples/ppo_trainer/run_webshop.sh
agent_system/environments/env_package/webshop/envs.py
agent_system/environments/env_package/webshop/projection.py
agent_system/environments/prompts/webshop.py
agent_system/environments/env_package/webshop/webshop/setup.sh
agent_system/environments/env_package/webshop/webshop/search_engine/run_indexing.sh
```

### 5.2 Setup and Data

`webshop/setup.sh` installs dependencies, downloads product data through
`gdown`, downloads spaCy models, converts product files, and builds Lucene
indexes through Pyserini.

Small data files:

```text
items_shuffle_1000.json
items_ins_v2_1000.json
```

Full data files:

```text
items_shuffle.json
items_ins_v2.json
```

Human instruction data:

```text
items_human_ins
```

Search index script:

```text
agent_system/environments/env_package/webshop/webshop/search_engine/run_indexing.sh
```

It builds multiple Lucene indexes:

```text
indexes_100
indexes
indexes_1k
indexes_100k
```

### 5.3 Environment Construction

Source: `make_envs`

If `config.env.webshop.use_small`:

```text
file_path = data/items_shuffle_1000.json
attr_path = data/items_ins_v2_1000.json
```

Otherwise:

```text
file_path = data/items_shuffle.json
attr_path = data/items_ins_v2.json
```

Env kwargs:

```text
observation_mode: "text"
num_products: None
human_goals: config.env.webshop.human_goals
file_path
attr_path
```

### 5.4 Reset

`WebshopMultiProcessEnv.reset` samples goal/session ids:

```text
validation: range(500)
train: range(500, len(goals))
```

`WebshopWorker.reset(idx)` calls:

```text
obs, info = self.env.reset(session=idx)
info["available_actions"] = self.env.get_available_actions()
info["won"] = False
```

### 5.5 Action Projection and Step

Source: `webshop_projection`

Expected model response:

```text
<think>...</think><action>search[...] or click[...]</action>
```

Projection lowercases the response and extracts the `<action>` span. It marks
the action invalid if:

```text
missing action tags
missing think tags
contains Chinese characters
```

`WebshopWorker.step(action)` calls:

```text
obs, reward, done, info = self.env.step(action)
```

Then enriches:

```text
info["available_actions"] = self.env.get_available_actions()
info["task_score"] = reward
info["won"] = done and reward == 1.0
reward = 10.0 if won else 0.0
```

The manager adds:

```text
is_action_valid
raw_text_action
projected_action
```

### 5.6 Prompt and History

Source: `WebshopEnvironmentManager.build_text_obs`

The initial prompt includes:

```text
task_description
current_observation
available_actions
```

Later prompts include bounded memory:

```text
step_count
history_length
action_history
current_step
current_observation
available_actions
```

If prompt length exceeds 13,000 characters, it falls back to no-history prompt.

### 5.7 WebShop Gaps

Not yet validated:

```text
whether setup.sh works in the server conda env
whether Google Drive downloads are accessible
whether Java/OpenJDK/Pyserini index build works
which index path WebAgentTextEnv selects at runtime
whether available_actions is sufficient for residual step typing
whether page type or selected product can be exported from info
```

Skill-OPD likely needs WebShop-specific fields:

```text
task_description
available_actions
page observation
page/action type
projected_action
task_score
won
selected product / final buy action, if accessible
```

## 6. Hypothetical Sokoban Example

This is a target shape for checking real exported JSONL.

```json
{
  "parquet_row": {
    "data_source": "text",
    "prompt": [{"role": "user", "content": ""}],
    "ability": "agent",
    "extra_info": {"split": "test", "index": 0}
  },
  "env_reset_obs": {
    "text": "You are playing Sokoban... Current observation: ...",
    "image": null,
    "anchor": "initial_room_state"
  },
  "model_input_step_0": {
    "raw_prompt_ids": "[chat-template token ids]",
    "input_ids": "[left-padded prompt token ids]"
  },
  "model_output_step_0": {
    "responses": "[response token ids]",
    "response_text": "<think>...</think><action>up</action>",
    "rollout_log_probs": "[optional per-token vLLM log probs]"
  },
  "env_step_0": {
    "text_action": "<think>...</think><action>up</action>",
    "projected_action": 1,
    "reward": 0.0,
    "done": false,
    "info": {
      "is_action_valid": true,
      "raw_text_action": "<think>...</think><action>up</action>",
      "projected_action": 1,
      "won": false
    }
  },
  "loss_view": {
    "trained_tokens": "response tokens only",
    "not_trained_tokens": "prompt and environment observation tokens",
    "response_mask_source": "attention_mask[:, -response_length:]",
    "advantage_source": "GRPO/GiGPO/PPO from token_level_rewards"
  }
}
```

## 7. Skill-OPD Field Readiness

| Field | Needed by Skill-OPD | Current source | Status |
|---|---|---|---|
| trajectory_id | yes | `traj_uid` | available |
| env_name | yes | config | available |
| step_id | yes | rollout step index | available |
| prompt_text / h_t | yes | decoded `prompts` or raw prompt | available, must validate quality |
| response_text | yes | decoded `responses` | available |
| response_token_ids | yes | `responses` | available |
| response_mask | v1.1 useful | derived from `attention_mask` and `responses` | added to exporter |
| rollout_log_probs | v1.1 useful | vLLM output | added if present |
| text_action | yes | decoded model response | added |
| projected_action | yes | EnvManager projection | added |
| reward | yes | env step reward | available |
| done | yes | env step done | added |
| success | yes | `success_evaluator` | available |
| is_action_valid | yes | projection validity | available |
| info | yes | env info | available |
| old_log_probs | later distill/RL analysis | trainer recompute | not exported in phase 1 |
| ref_log_probs | later distill/RL analysis | trainer ref policy | not exported in phase 1 |
| advantages | later RL analysis | trainer advantage computation | not exported in phase 1 |
| teacher logits | teacher scoring | not in rollout | intentionally deferred |
| global/residual skill | Skill-OPD method | not in rollout | intentionally deferred |

## 8. Immediate Execution Checklist

1. Run Sokoban validation-only rollout on GPU server.
2. Inspect `outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl`.
3. Confirm the following are non-empty in real output:

```text
steps[].prompt_text
steps[].response_text
steps[].response_token_ids
steps[].text_action
steps[].projected_action
steps[].reward
steps[].done
steps[].is_action_valid
steps[].info
```

4. If `prompt_text` is not the full current `h_t`, inspect `raw_prompt` and
   `prompts` in `available_keys`.
5. If `rollout_log_probs` is missing, confirm whether vLLM config returns
   logprobs in this environment.
6. Do not start AppWorld/WebShop training until Sokoban export is validated.

