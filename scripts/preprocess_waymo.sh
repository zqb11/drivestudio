#!/usr/bin/env bash
python datasets/tools/humanpose_process.py \
    --dataset waymo \
    --data_root data/waymo/processed/training \
    --scene_ids 23 \
    --save_temp --verbose