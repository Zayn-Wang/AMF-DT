#!/bin/bash

cd "$(dirname "$0")/.."

python -m inference.run_subnet5 \
  --gpu 1 \
  --val_batch 4 \
  --num_workers_val 10 \
  --best_model_name multitask_subnet5 \
  --test_dir "../samples/Input_Test/" \
  --pfs_model_path "../weights/multitask_subnet5_PFS.pth" \
  --os_model_path "../weights/multitask_subnet5_OS.pth"
