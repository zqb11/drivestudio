#!/usr/bin/env bash
python datasets/preprocess.py \
    --data_root data/nuscenes/raw/v1.0-mini  \
    --target_dir data/nuscenes/processed \
    --dataset nuscenes \
    --split v1.0-mini \
    --start_idx 0 \
    --num_scenes 10 \
    --interpolate_N 4 \
    --workers 32 \
    --process_keys images lidar calib dynamic_masks objects