#!/bin/bash

cd "$(dirname "$0")/.."

python -m inference.eval_subnet1 \
  --gpu 1 \
  --val_batch 16 \
  --num_workers_val 10 \
  --best_model_name multitask_subnet1 \
  --train_dir "../samples/Input_Train/" \
  --val_dir "../samples/Input_Val/" \
  --test_dir "../samples/Input_Test/" \
  --train_csv "../samples/train_events.csv" \
  --val_csv "../samples/valid_events.csv" \
  --test_csv "../samples/test_events.csv" \
  --pfs_model_path "../weights/multitask_subnet1_PFS.pth" \
  --os_model_path "../weights/multitask_subnet1_OS.pth"
