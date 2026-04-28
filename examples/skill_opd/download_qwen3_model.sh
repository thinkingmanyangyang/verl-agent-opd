#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B}
MODEL_DIR=${MODEL_DIR:-"$HOME/models/${MODEL_NAME//\//_}"}

python3 - <<PY
from huggingface_hub import snapshot_download

model_name = "${MODEL_NAME}"
model_dir = "${MODEL_DIR}"
print(f"Downloading {model_name} to {model_dir}")
snapshot_download(repo_id=model_name, local_dir=model_dir)
print(model_dir)
PY

