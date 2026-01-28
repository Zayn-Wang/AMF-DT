#!/bin/bash

cd "$(dirname "$0")/.."

python -m torch.distributed.run --nproc_per_node=6 -m main.train_subnet1 \
  --gpu 0,1,2,3,4,5 \
  --lr 0.0001 \
  --lr_decay 0.15 \
  --rand_p 0.45 \
  --max_epochs 250 \
  --train_batch 4 \
  --val_batch 8 \
  --test_batch 8 \
  --skip_epoch_model 25 \
  --best_model_name multitask_subnet1 \
  --train_dir "../samples/Input_Train/" \
  --val_dir "../samples/Input_Val/" \
  --test_dir "../samples/Input_Test/" \
  --train_csv "../samples/train_events.csv" \
  --val_csv "../samples/valid_events.csv" \
  --test_csv "../samples/test_events.csv" \
#  --verbose  # --verbose enables verbose mode; remove this flag to disable it.
