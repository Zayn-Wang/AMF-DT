#!/bin/bash

cd "$(dirname "$0")/.."

python -m inference.run_subnet3 \
  --gpu 1 \
  --val_batch 40 \
  --num_workers_val 10 \
  --best_model_name multitask_subnet3 \
  --test_dir "../samples/Input_Test/" \
  --pfs_model_path "../weights/multitask_subnet3_PFS.pth" \
  --os_model_path "../weights/multitask_subnet3_OS.pth"
