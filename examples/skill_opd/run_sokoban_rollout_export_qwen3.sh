#!/usr/bin/env bash
set -euo pipefail
set -x

# This script follows examples/gigpo_trainer/run_sokoban.sh but switches to
# text observations and enables the Skill-OPD offline rollout exporter.
# It runs validation-only rollout export.  It still requires a GPU/vLLM setup,
# so do not run it on the current CPU-only machine.

ENGINE=${ENGINE:-vllm}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B}
DATA_ROOT=${DATA_ROOT:-"$HOME/data/verl-agent"}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-8}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
GROUP_SIZE=${GROUP_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-15}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
NUM_GPUS=${NUM_GPUS:-1}
TP_SIZE=${TP_SIZE:-1}
EXPORT_PATH=${EXPORT_PATH:-outputs/skill_opd/rollouts/sokoban_qwen3.jsonl}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}

MODE=text \
TRAIN_DATA_SIZE="$TRAIN_DATA_SIZE" \
VAL_DATA_SIZE="$VAL_DATA_SIZE" \
LOCAL_DIR="$DATA_ROOT" \
bash examples/skill_opd/prepare_sokoban_data.sh

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$DATA_ROOT/text/train.parquet" \
  data.val_files="$DATA_ROOT/text/test.parquet" \
  data.train_batch_size="$TRAIN_DATA_SIZE" \
  data.val_batch_size="$VAL_DATA_SIZE" \
  data.max_prompt_length=2048 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_DATA_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$TP_SIZE" \
  actor_rollout_ref.rollout.name="$ENGINE" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=0.95 \
  env.env_name=Sokoban \
  env.seed=0 \
  env.max_steps="$MAX_STEPS" \
  env.rollout.n="$GROUP_SIZE" \
  env.sokoban.mode='tiny_rgb_array' \
  env.resources_per_worker.num_cpus="$NUM_CPUS_PER_ENV_WORKER" \
  trainer.critic_warmup=0 \
  "trainer.logger=['console']" \
  trainer.project_name='skill_opd_sokoban' \
  trainer.experiment_name='qwen3_rollout_export' \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  +skill_opd.export_rollouts=True \
  +skill_opd.export_path="$EXPORT_PATH" \
  +skill_opd.overwrite=True \
  "$@"
