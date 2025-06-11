#!/bin/bash
python -m pip install -r requirements.txt

pip install -U sentence-transformers
pip install -U datasets

python me5_large_instruct.py \
  --hf_token "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" \
  --wandb_api_key "2570172483ba90dcd524a971e6a6efe6aa0f6581" \
  --batch_size 16 \
  --epoch 10
