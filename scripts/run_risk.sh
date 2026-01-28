#!/bin/bash

cd "$(dirname "$0")/.."

python -m inference.run_subnet1 \
  --gpu 1 \
  --val_batch 16 \
  --num_workers_val 10 \
  --best_model_name multitask_subnet1 \
  --test_dir "../samples/Input_Test/" \
  --pfs_model_path "../weights/multitask_subnet1_PFS.pth" \
  --os_model_path "../weights/multitask_subnet1_OS.pth"
