#!/usr/bin/env bash
set -euo pipefail

MODE=${MODE:-text}
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-8}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-8}
LOCAL_DIR=${LOCAL_DIR:-"$HOME/data/verl-agent"}

python3 -m examples.data_preprocess.prepare \
  --mode "$MODE" \
  --local_dir "$LOCAL_DIR" \
  --train_data_size "$TRAIN_DATA_SIZE" \
  --val_data_size "$VAL_DATA_SIZE"

echo "Prepared Sokoban placeholder data under $LOCAL_DIR/$MODE"

