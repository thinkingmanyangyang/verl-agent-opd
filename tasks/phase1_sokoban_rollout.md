# Phase 1 Server Task: Sokoban Offline Rollout Export

Objective: generate a small validation-only Sokoban rollout file using Qwen3 and
the Skill-OPD exporter.

Assumption: the server already has a working `verl-agent` conda environment with
GPU, Ray, vLLM, PyTorch, datasets, and Hugging Face dependencies.

## 1. Clone or Update Repository

Fresh clone:

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:thinkingmanyangyang/verl-agent-opd.git
cd verl-agent-opd
```

Existing clone:

```bash
cd ~/projects/verl-agent-opd
git checkout main
git pull --rebase origin main
```

## 2. Activate Environment and Check Runtime

```bash
conda activate verl-agent

python -V
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
```

Static checks:

```bash
python -m py_compile \
  agent_system/skill_opd/config.py \
  agent_system/skill_opd/io.py \
  agent_system/skill_opd/schema.py \
  agent_system/skill_opd/rollout_exporter.py \
  agent_system/skill_opd/rollout_hook.py \
  agent_system/multi_turn_rollout/rollout_loop.py

bash -n examples/skill_opd/*.sh
```

## 3. Prepare Sokoban Placeholder Data

verl-agent uses parquet data here to control batch size and modality. Sokoban
states are produced by the environment during rollout.

```bash
MODE=text \
TRAIN_DATA_SIZE=8 \
VAL_DATA_SIZE=8 \
LOCAL_DIR=$HOME/data/verl-agent \
bash examples/skill_opd/prepare_sokoban_data.sh
```

Expected files:

```text
$HOME/data/verl-agent/text/train.parquet
$HOME/data/verl-agent/text/test.parquet
```

## 4. Download Qwen3

Start with 4B.

```bash
MODEL_NAME=Qwen/Qwen3-4B \
MODEL_DIR=$HOME/models/Qwen3-4B \
bash examples/skill_opd/download_qwen3_model.sh
```

If Hugging Face is slow:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Run 8B only after 4B works.

```bash
MODEL_NAME=Qwen/Qwen3-8B \
MODEL_DIR=$HOME/models/Qwen3-8B \
bash examples/skill_opd/download_qwen3_model.sh
```

## 5. Run Validation-Only Rollout Export

Single-GPU 4B smoke run:

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

The script sets:

```text
trainer.val_only=True
skill_opd.export_rollouts=True
```

It should export validation rollouts without updating model parameters.

## 6. Inspect Output

```bash
ls -lh outputs/skill_opd/rollouts/
wc -l outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl
head -n 1 outputs/skill_opd/rollouts/sokoban_qwen3_4b.jsonl
```

Expected per-trajectory fields:

```text
trajectory_id
env_name
total_reward
episode_length
success
tool_call_count
steps
```

Expected per-step fields:

```text
step_id
prompt_text
response_text
response_token_ids
reward
done
active_mask
is_action_valid
info
available_keys
```

## 7. Report Back

Record the following in `tasks/backlog.md` or a new run note:

- Exact command.
- GPU type and count.
- Model path.
- Number of exported trajectories.
- First failure if the run fails.
- Whether `prompt_text`, `response_text`, and `response_token_ids` are present.

