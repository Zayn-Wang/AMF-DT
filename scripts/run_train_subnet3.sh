#!/bin/bash

cd "$(dirname "$0")/.."

python -m torch.distributed.run --nproc_per_node=5 -m main.train_subnet3 \
  --gpu 1,2,3,4,5 \
  --lr 0.00001 \
  --lr_decay 0.1 \
  --rand_p 0.35 \
  --max_epochs 250 \
  --train_batch 16 \
  --val_batch 16 \
  --test_batch 16 \
  --skip_epoch_model 40 \
  --best_model_name multitask_subnet3 \
  --train_dir "../samples/Input_Train/" \
  --val_dir "../samples/Input_Val/" \
  --test_dir "../samples/Input_Test/" \
  --train_csv "../samples/train_events.csv" \
  --val_csv "../samples/valid_events.csv" \
  --test_csv "../samples/test_events.csv" \
#  --verbose  # uncomment to enable verbose mode
