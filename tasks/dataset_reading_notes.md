# Dataset and Rollout Reading Notes

Last updated: 2026-05-07

This file is a working reading log for the code paths relevant to
`Sokoban`, `AppWorld`, `WebShop`, and the Skill-OPD offline rollout exporter.
It is intentionally file-centric so that another engineer can continue the
audit without repeating the initial repository search.

## 1. Repository-Level Files

### README.md

Role:

```text
Project-level overview for verl-agent.
```

Important content:

```text
Points users to supported agentic environments, trainers, and examples.
Useful as a map, not as an exact data-flow reference.
```

Key fields:

```text
No runtime fields.
```

Unresolved questions:

```text
README does not specify Skill-OPD exporter behavior because that is our local extension.
```

Next file if unclear:

```text
examples/*/run_*.sh
verl/trainer/main_ppo.py
```

### tasks/README.md

Role:

```text
Local project task index for Skill-OPD work.
```

Important content:

```text
Documents the active repository, workflow, and task file meanings.
```

Key fields:

```text
No runtime fields.
```

Unresolved questions:

```text
Needs to include every new audit file added by this round.
```

Next file if unclear:

```text
tasks/roadmap.md
tasks/phase1_sokoban_rollout.md
```

## 2. Dataset and DataProto

### examples/data_preprocess/prepare.py

Role:

```text
Generic data preparation entrypoint.
```

Important classes/functions:

```text
main
dataset-specific prepare helpers if configured
```

Key fields:

```text
prompt
data_source
extra_info
train_files
val_files
```

Unresolved questions:

```text
Sokoban primarily uses env-generated states, so this file is not the main source
of Sokoban board states. Need server run to confirm exact parquet files used by
the selected script/config.
```

Next file if unclear:

```text
examples/gigpo_trainer/run_sokoban.sh
examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

### verl/utils/dataset/rl_dataset.py

Role:

```text
Converts parquet rows into tokenized prompt samples consumed by the trainer.
```

Important classes/functions:

```text
RLHFDataset
RLHFDataset._download
RLHFDataset._read_files_and_tokenize
RLHFDataset.__getitem__
collate_fn
process_image
process_video
```

Key fields:

```text
input_ids
attention_mask
position_ids
raw_prompt_ids
raw_prompt
full_prompts
data_source
extra_info
index
tools_kwargs
```

Unresolved questions:

```text
Need to confirm whether the chosen Sokoban script sets return_raw_chat or
return_full_prompt. The exporter can still decode rollout prompts from token ids,
so this is not blocking.
```

Next file if unclear:

```text
verl/trainer/config/ppo_trainer.yaml
verl/trainer/ppo/ray_trainer.py
```

### verl/protocol.py

Role:

```text
Defines DataProto, the central tensor + metadata batch object used across verl.
```

Important classes/functions:

```text
DataProto
DataProto.from_single_dict
DataProto.pop
DataProto.union
DataProto.repeat
DataProtoItem
```

Key fields:

```text
batch
non_tensor_batch
meta_info
```

Unresolved questions:

```text
No major ambiguity. The main implementation constraint is that non-tensor fields
must be numpy arrays, which is why rollout/export metadata needs careful packing.
```

Next file if unclear:

```text
verl/trainer/ppo/ray_trainer.py
agent_system/multi_turn_rollout/rollout_loop.py
```

## 3. Trainer Entry and Main Loop

### verl/trainer/main_ppo.py

Role:

```text
Hydra/Ray entrypoint that builds envs, tokenizer, workers, datasets, reward
manager, trajectory collector, and RayPPOTrainer.
```

Important classes/functions:

```text
main
run_ppo
TaskRunner.run
create_rl_dataset
create_rl_sampler
```

Key fields/config:

```text
config.actor_rollout_ref.model.path
config.env.rollout.n
config.reward_model.reward_manager
config.data.train_files
config.data.val_files
```

Unresolved questions:

```text
Need server config dump from the actual Sokoban run to confirm exact Hydra
overrides and whether validation-only mode is enabled as intended.
```

Next file if unclear:

```text
verl/trainer/ppo/ray_trainer.py
agent_system/environments/env_manager.py
```

### verl/trainer/config/ppo_trainer.yaml

Role:

```text
Default trainer config.
```

Important classes/functions:

```text
Hydra config only.
```

Key fields/config:

```text
data.*
actor_rollout_ref.*
algorithm.*
trainer.*
env.*
reward_model.*
```

Unresolved questions:

```text
Exact active values depend on the shell script overrides. Do not infer final
runtime behavior from defaults alone.
```

Next file if unclear:

```text
examples/gigpo_trainer/run_sokoban.sh
examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

### verl/trainer/ppo/ray_trainer.py

Role:

```text
Main PPO/GRPO/GiGPO orchestration loop.
```

Important classes/functions:

```text
RayPPOTrainer
RayPPOTrainer._create_dataloader
RayPPOTrainer._validate
RayPPOTrainer.fit
compute_response_mask
apply_invalid_action_penalty
compute_advantage
apply_kl_penalty
```

Key fields:

```text
gen_batch
response_mask
old_log_probs
ref_log_prob
rollout_log_probs
token_level_scores
token_level_rewards
advantages
returns
uid
traj_uid
step_rewards
anchor_obs
is_action_valid
```

Unresolved questions:

```text
Need a real run to confirm whether multi_turn loss_mask is active in the selected
config and whether rollout_log_probs is present for the chosen vLLM settings.
```

Next file if unclear:

```text
agent_system/multi_turn_rollout/rollout_loop.py
verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py
```

### verl/trainer/ppo/reward.py

Role:

```text
Wrapper for reward function execution.
```

Important classes/functions:

```text
compute_reward
compute_reward_async
```

Key fields:

```text
token_level_scores
reward_extra_infos_dict
```

Unresolved questions:

```text
No major ambiguity for episode reward path. Non-episode reward managers are not
the current Skill-OPD focus.
```

Next file if unclear:

```text
agent_system/reward_manager/episode.py
```

### agent_system/reward_manager/episode.py

Role:

```text
Converts episode rewards from env rollout into token-level reward tensors.
```

Important classes/functions:

```text
EpisodeRewardManager
EpisodeRewardManager.__call__
```

Key fields:

```text
episode_rewards
episode_lengths
responses
attention_mask
reward_tensor
valid_response_length
```

Unresolved questions:

```text
Need server inspection to confirm exact episode_rewards shape after Sokoban
rollout flattening.
```

Next file if unclear:

```text
agent_system/multi_turn_rollout/rollout_loop.py
```

### verl/trainer/ppo/core_algos.py

Role:

```text
Core RL math for advantage, KL, and actor loss.
```

Important classes/functions:

```text
compute_grpo_outcome_advantage
compute_policy_loss
compute_kl
masked_mean
```

Key fields:

```text
old_log_probs
log_prob
advantages
response_mask
token_level_rewards
uid
traj_uid
```

Unresolved questions:

```text
Future distillation loss will need a clear hook near actor loss, but that should
wait until teacher target tensors are specified and validated.
```

Next file if unclear:

```text
verl/workers/* actor update implementation
```

### gigpo/core_gigpo.py

Role:

```text
GiGPO-specific step return and advantage logic.
```

Important classes/functions:

```text
compute_step_discounted_returns
compute_gigpo_outcome_advantage
```

Key fields:

```text
step_rewards
anchor_obs
uid
traj_uid
token_level_rewards
response_mask
```

Unresolved questions:

```text
Need to confirm whether Sokoban first-stage server run uses GRPO or GiGPO. The
exporter itself is independent, but interpretation of step_rewards is not.
```

Next file if unclear:

```text
examples/gigpo_trainer/run_sokoban.sh
examples/grpo_trainer/run_sokoban.sh
```

## 4. Rollout and Export

### agent_system/multi_turn_rollout/rollout_loop.py

Role:

```text
Central agentic multi-turn rollout loop.
```

Important classes/functions:

```text
TrajectoryCollector
TrajectoryCollector.preprocess_single_sample
TrajectoryCollector.preprocess_batch
TrajectoryCollector.vanilla_multi_turn_loop
TrajectoryCollector.dynamic_multi_turn_loop
TrajectoryCollector.gather_rollout_data
TrajectoryCollector.multi_turn_loop
```

Key fields:

```text
batch_input
text_actions
rewards
dones
infos
active_masks
total_batch_list
total_infos
episode_rewards
episode_lengths
success
is_action_valid
text_action
projected_action
rollout_log_probs
```

Unresolved questions:

```text
Need real rollout output to verify total_infos nesting and whether every step has
projected_action for Sokoban.
```

Next file if unclear:

```text
agent_system/environments/env_manager.py
agent_system/skill_opd/rollout_exporter.py
```

### agent_system/multi_turn_rollout/utils.py

Role:

```text
Utility helpers for rollout batch conversion and token/mask handling.
```

Important classes/functions:

```text
to_list_of_dict
helper functions used by rollout_loop
```

Key fields:

```text
DataProto item fields
tensor/non-tensor per-sample dictionaries
```

Unresolved questions:

```text
Need to inspect again if exporter shape bugs appear on server.
```

Next file if unclear:

```text
agent_system/skill_opd/rollout_exporter.py
```

### verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py

Role:

```text
vLLM rollout worker implementation that generates response tensors and optional
rollout log probabilities.
```

Important classes/functions:

```text
vLLMRollout
vLLMRollout.generate_sequences
```

Key fields:

```text
prompts
responses
input_ids
attention_mask
position_ids
rollout_log_probs
```

Unresolved questions:

```text
Need server run to confirm whether rollout_log_probs is emitted under the active
generation config.
```

Next file if unclear:

```text
verl/trainer/ppo/ray_trainer.py diagnostics around rollout_log_probs
```

### agent_system/skill_opd/config.py

Role:

```text
Small config helpers for enabling/disabling Skill-OPD export.
```

Important classes/functions:

```text
SkillOPDExportConfig or related config parsing helpers
```

Key fields:

```text
enabled
path
```

Unresolved questions:

```text
Need server config dump to ensure Hydra override names are ergonomic enough.
```

Next file if unclear:

```text
examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

### agent_system/skill_opd/io.py

Role:

```text
JSONL writing and filesystem helper layer.
```

Important classes/functions:

```text
JSONL writer helpers
```

Key fields:

```text
export path
record dict
```

Unresolved questions:

```text
Need to verify behavior when multiple Ray workers write concurrently. Current
first-stage assumption is validation/small rollout, not high-concurrency export.
```

Next file if unclear:

```text
agent_system/skill_opd/rollout_hook.py
```

### agent_system/skill_opd/schema.py

Role:

```text
Defines serializable trajectory and step records for offline Skill-OPD data.
```

Important classes/functions:

```text
RolloutStepRecord
RolloutTrajectoryRecord
```

Key fields:

```text
trajectory_id
env_name
step_id
prompt_text
response_text
response_token_ids
text_action
projected_action
reward
done
is_action_valid
info
available_keys
response_mask
rollout_log_probs
```

Unresolved questions:

```text
Need real JSONL to confirm which optional fields are actually non-empty.
```

Next file if unclear:

```text
agent_system/skill_opd/rollout_exporter.py
```

### agent_system/skill_opd/rollout_exporter.py

Role:

```text
Builds nested JSONL trajectory records from total_batch_list and total_infos.
```

Important classes/functions:

```text
RolloutExporter
RolloutExporter.build_records
```

Key fields:

```text
total_batch_list
total_infos
episode_rewards
episode_lengths
success
prompt_text
response_text
response_token_ids
response_mask
rollout_log_probs
done
metadata.done_is_inferred_from_episode_length
```

Unresolved questions:

```text
Fake exporter passes locally. Real server rollout must confirm token decoding and
field alignment across batch/step dimensions.
```

Next file if unclear:

```text
agent_system/multi_turn_rollout/rollout_loop.py
```

### agent_system/skill_opd/rollout_hook.py

Role:

```text
Small read-only hook called after multi-turn rollout to write JSONL if enabled.
```

Important classes/functions:

```text
maybe_export_rollout
```

Key fields:

```text
config
tokenizer
step_id
batch
obs
next_obs
text_actions
rewards
dones
infos
active_masks
total_batch_list
total_infos
```

Unresolved questions:

```text
Need server validation to confirm hook placement does not produce duplicate
records across validation/train loops.
```

Next file if unclear:

```text
examples/skill_opd/run_sokoban_rollout_export_qwen3.sh
```

## 5. Shared Environment Manager

### agent_system/environments/env_manager.py

Role:

```text
Builds env managers and translates between env reset/step outputs, prompts,
memory, and model actions.
```

Important classes/functions:

```text
make_envs
BaseEnvironmentManager
SokobanEnvironmentManager
WebshopEnvironmentManager
AppWorldEnvironmentManager
AlfWorldEnvironmentManager
success_evaluator methods
```

Key fields:

```text
obs
text_obs
anchor_obs
images
text_actions
projected_action
raw_text_action
is_action_valid
available_actions
tasks
supervisors
memory
history
```

Unresolved questions:

```text
Need server validation for exact shape/list nesting of info dicts after Ray env
parallelism.
```

Next file if unclear:

```text
agent_system/environments/env_package/<dataset>/*
agent_system/environments/prompts/<dataset>.py
```

### agent_system/memory/memory.py

Role:

```text
Stores step history used by EnvManager prompt builders.
```

Important classes/functions:

```text
Memory
memory.store or equivalent update methods
```

Key fields:

```text
previous observation
action
execution result
history text
```

Unresolved questions:

```text
Need to decide whether raw memory context should become a v1.1 export field.
For v1, prompt_text is sufficient h_t.
```

Next file if unclear:

```text
agent_system/environments/prompts/*.py
```

## 6. Sokoban Files

### examples/skill_opd/run_sokoban_rollout_export_qwen3.sh

Role:

```text
First-stage server script for exporting Sokoban offline rollouts with Qwen3.
```

Important classes/functions:

```text
Shell/Hydra overrides only.
```

Key fields/config:

```text
MODEL_PATH
DATA_ROOT
TRAIN_DATA_SIZE
VAL_DATA_SIZE
GROUP_SIZE
MAX_STEPS
NUM_GPUS
TP_SIZE
EXPORT_PATH
trainer.val_only
skill_opd.export.*
```

Unresolved questions:

```text
Needs server execution. Local CPU can only syntax-check this script.
```

Next file if unclear:

```text
examples/gigpo_trainer/run_sokoban.sh
examples/grpo_trainer/run_sokoban.sh
```

### examples/gigpo_trainer/run_sokoban.sh

Role:

```text
Reference Sokoban GiGPO training script.
```

Important classes/functions:

```text
Shell/Hydra overrides only.
```

Key fields/config:

```text
env.name=sokoban
env.rollout.n
max_steps
algorithm advantage estimator
batch sizes
```

Unresolved questions:

```text
Need compare exact overrides against our export script before server run.
```

Next file if unclear:

```text
verl/trainer/main_ppo.py
verl/trainer/ppo/ray_trainer.py
```

### examples/grpo_trainer/run_sokoban.sh

Role:

```text
Reference Sokoban GRPO training script.
```

Important classes/functions:

```text
Shell/Hydra overrides only.
```

Key fields/config:

```text
env.name=sokoban
GRPO estimator settings
batch sizes
rollout settings
```

Unresolved questions:

```text
Need decide whether first Skill-OPD experiment should inherit GRPO or GiGPO
settings after offline export is validated.
```

Next file if unclear:

```text
verl/trainer/ppo/core_algos.py
gigpo/core_gigpo.py
```

### agent_system/environments/env_package/sokoban/envs.py

Role:

```text
Multiprocess/Ray wrapper around SokobanEnv.
```

Important classes/functions:

```text
SokobanWorker
SokobanMultiProcessEnv
build_sokoban_envs
reset
step
```

Key fields:

```text
seed
group_n
mode
dim_room
num_boxes
max_steps
search_depth
obs
reward
done
info
```

Unresolved questions:

```text
Need real validation to confirm reset seeds and group_n produce expected repeated
same-level rollouts for grouping.
```

Next file if unclear:

```text
agent_system/environments/env_package/sokoban/sokoban/env.py
```

### agent_system/environments/env_package/sokoban/projection.py

Role:

```text
Parses model text response into Sokoban action id.
```

Important classes/functions:

```text
sokoban_projection
```

Key fields:

```text
<think>
<action>
up
down
left
right
still
valids
```

Unresolved questions:

```text
No major ambiguity. Invalid outputs are expected and should be exported.
```

Next file if unclear:

```text
agent_system/environments/env_manager.py::SokobanEnvironmentManager.step
```

### agent_system/environments/env_package/sokoban/sokoban/env.py

Role:

```text
Concrete Sokoban environment implementation.
```

Important classes/functions:

```text
SokobanEnv
SokobanEnv.reset
SokobanEnv.step
```

Key fields:

```text
room_state
reward
done
action_is_effective
won
max_steps
```

Unresolved questions:

```text
Need inspect real info dict from first server rollout to confirm all useful keys.
```

Next file if unclear:

```text
agent_system/environments/env_package/sokoban/sokoban/base.py
agent_system/environments/env_package/sokoban/sokoban/room_utils.py
```

### agent_system/environments/env_package/sokoban/sokoban/base.py

Role:

```text
Base Sokoban mechanics inherited by SokobanEnv.
```

Important classes/functions:

```text
GymSokobanEnv or related base classes
```

Key fields:

```text
room_state
player position
box/target state
render modes
```

Unresolved questions:

```text
Only needed if reward/done behavior is suspicious in server rollout.
```

Next file if unclear:

```text
agent_system/environments/env_package/sokoban/sokoban/env.py
```

### agent_system/environments/env_package/sokoban/sokoban/room_utils.py

Role:

```text
Room generation utilities.
```

Important classes/functions:

```text
generate_room
```

Key fields:

```text
dim_room
num_steps
num_boxes
search_depth
seed
```

Unresolved questions:

```text
Potential source of slow reset if search_depth is high. Not blocking for audit.
```

Next file if unclear:

```text
examples/gigpo_trainer/run_sokoban.sh
```

### agent_system/environments/prompts/sokoban.py

Role:

```text
Sokoban prompt templates and text observation construction.
```

Important classes/functions:

```text
build_text_obs
prompt template helpers
```

Key fields:

```text
board text
history
action format
available action names
```

Unresolved questions:

```text
Need inspect exported prompt_text to confirm h_t has enough history for teacher
scoring without exporting memory_context separately.
```

Next file if unclear:

```text
agent_system/environments/env_manager.py::SokobanEnvironmentManager
```

## 7. AppWorld Files

### examples/env_server/start_appworld_server.sh

Role:

```text
Starts AppWorld service instances and writes available ports.
```

Important classes/functions:

```text
Shell script only.
```

Key fields/config:

```text
appworld_ports.ports
start port
number of instances
train_batch_size
group_size
val_batch_size
```

Unresolved questions:

```text
Need server-specific conda environment and port availability. Do not run locally.
```

Next file if unclear:

```text
agent_system/environments/env_package/appworld/envs.py
```

### agent_system/environments/env_package/appworld/envs.py

Role:

```text
Ray/multiprocess wrapper over remote AppWorld environments.
```

Important classes/functions:

```text
load_available_ports
AppWorldWorker
AppWorldWorker.reset
AppWorldWorker.step
AppWorldMultiProcessEnv
build_appworld_envs
```

Key fields:

```text
task_id
dataset_name
supervisor
instruction
port
action/code
execution result
won
step_count
reward
done
```

Unresolved questions:

```text
Current info appears to export success and step count, but not full API trace,
collateral damage, or detailed evaluation report. Skill-OPD AppWorld v1.1 should
add AppWorld-specific info fields after setup is validated.
```

Next file if unclear:

```text
agent_system/environments/env_package/appworld/projection.py
agent_system/environments/prompts/appworld.py
```

### agent_system/environments/env_package/appworld/projection.py

Role:

```text
Extracts Python code action from model response.
```

Important classes/functions:

```text
appworld_projection
```

Key fields:

```text
<think>
<code>
code text
valids
```

Unresolved questions:

```text
Need real rollout to inspect common invalid formats and whether fallback text
should be stored for debugging.
```

Next file if unclear:

```text
agent_system/environments/env_manager.py::AppWorldEnvironmentManager.step
```

### agent_system/environments/prompts/appworld.py

Role:

```text
AppWorld initial and history prompt templates.
```

Important classes/functions:

```text
prompt builder helpers
```

Key fields:

```text
instruction
supervisor
API guidance
Code k
Result k
```

Unresolved questions:

```text
Need decide whether supervisor should be considered part of q, privileged
context, or ordinary h_t for Skill-OPD teacher scoring.
```

Next file if unclear:

```text
docs/method design for Skill-OPD context definitions
```

## 8. WebShop Files

### examples/gigpo_trainer/run_webshop.sh

Role:

```text
Reference WebShop GiGPO training script.
```

Important classes/functions:

```text
Shell/Hydra overrides only.
```

Key fields/config:

```text
env.name=webshop
data/resource paths
rollout settings
batch sizes
```

Unresolved questions:

```text
Need verify resource paths after setup.sh on server.
```

Next file if unclear:

```text
agent_system/environments/env_package/webshop/webshop/setup.sh
```

### examples/grpo_trainer/run_webshop.sh

Role:

```text
Reference WebShop GRPO training script.
```

Important classes/functions:

```text
Shell/Hydra overrides only.
```

Key fields/config:

```text
env.name=webshop
advantage estimator settings
resource paths
```

Unresolved questions:

```text
Need decide later whether WebShop should use GRPO or GiGPO baseline.
```

Next file if unclear:

```text
verl/trainer/ppo/core_algos.py
gigpo/core_gigpo.py
```

### examples/ppo_trainer/run_webshop.sh

Role:

```text
Reference WebShop PPO script.
```

Important classes/functions:

```text
Shell/Hydra overrides only.
```

Key fields/config:

```text
PPO settings
WebShop env settings
```

Unresolved questions:

```text
Likely baseline-only reference. Not a Skill-OPD first-stage target.
```

Next file if unclear:

```text
examples/grpo_trainer/run_webshop.sh
```

### agent_system/environments/env_package/webshop/envs.py

Role:

```text
Multiprocess wrapper around WebAgentTextEnv.
```

Important classes/functions:

```text
WebshopWorker
WebshopMultiProcessEnv
build_webshop_envs
reset
step
extract_task
get_available_actions
```

Key fields:

```text
session
goal_idx
observation
available_actions
task_score
won
reward
done
info
```

Unresolved questions:

```text
Need setup validation because WebShop depends on downloaded product data and
Lucene/Pyserini indexes.
```

Next file if unclear:

```text
agent_system/environments/env_package/webshop/webshop/setup.sh
agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py
```

### agent_system/environments/env_package/webshop/projection.py

Role:

```text
Extracts WebShop action text from model response.
```

Important classes/functions:

```text
webshop_projection
```

Key fields:

```text
<think>
<action>
search[...]
click[...]
valids
```

Unresolved questions:

```text
Need inspect real invalid outputs and available action mismatch frequency.
```

Next file if unclear:

```text
agent_system/environments/env_manager.py::WebshopEnvironmentManager.step
```

### agent_system/environments/prompts/webshop.py

Role:

```text
WebShop prompt templates.
```

Important classes/functions:

```text
prompt builder helpers
```

Key fields:

```text
task
page observation
available actions
search/click/buy format
history
```

Unresolved questions:

```text
Need inspect exported prompt_text after setup to confirm available actions are
always visible in h_t.
```

Next file if unclear:

```text
agent_system/environments/env_manager.py::WebshopEnvironmentManager
```

### agent_system/environments/env_package/webshop/webshop/setup.sh

Role:

```text
Downloads and prepares WebShop data/resources.
```

Important classes/functions:

```text
Shell setup only.
```

Key fields/config:

```text
gdown files
requirements
OpenJDK
faiss-cpu
spaCy models
resource files
```

Unresolved questions:

```text
Potential setup fragility: Google Drive availability, Java version, Pyserini,
index paths. Should be validated on server before planning WebShop experiments.
```

Next file if unclear:

```text
agent_system/environments/env_package/webshop/webshop/search_engine/run_indexing.sh
```

### agent_system/environments/env_package/webshop/webshop/search_engine/run_indexing.sh

Role:

```text
Builds Lucene indexes for WebShop search.
```

Important classes/functions:

```text
Shell indexing only.
```

Key fields/config:

```text
resources_100
resources
resources_1k
resources_100k
index path
```

Unresolved questions:

```text
Need decide which resource size is acceptable for first WebShop prototype.
```

Next file if unclear:

```text
examples/gigpo_trainer/run_webshop.sh
```

### agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py

Role:

```text
Underlying text-mode WebShop environment.
```

Important classes/functions:

```text
WebAgentTextEnv
WebAgentTextEnv.reset
WebAgentTextEnv.step
```

Key fields:

```text
observation
reward
done
info
search
click
buy
task_score
```

Unresolved questions:

```text
Need inspect actual info dict on server after setup. This is where page-level
and score-level debugging fields should come from.
```

Next file if unclear:

```text
agent_system/environments/env_package/webshop/envs.py
```

## 9. Immediate Reading Conclusions

Confirmed locally:

```text
Sokoban has the most complete path for first-stage offline rollout export.
AppWorld has a code path but needs external server process validation.
WebShop has a code path but needs data/index setup validation.
The trainer computes response_mask after rollout, so exporter deriving it from
attention_mask is acceptable for v1.
old_log_probs/ref_log_prob/advantages are trainer-side fields and should not be
required from the v1 rollout exporter.
```

Needs server validation:

```text
Sokoban JSONL real shape and non-empty required fields.
Whether rollout_log_probs is emitted by the active vLLM config.
Whether text_action/projected_action/dones align with step order after Ray env
parallelism.
Whether prompt_text fully captures h_t for teacher scoring.
```

Do not change yet:

```text
actor loss
advantage algorithm
teacher scoring
skill retrieval
AppWorld/WebShop training scripts
```

