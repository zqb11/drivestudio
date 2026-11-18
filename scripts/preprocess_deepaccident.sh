#!/usr/bin/env bash
# python datasets/preprocess.py \
#     --data_root /mnt/sda1/data/TA/deepaccident/raw/mini/type1_subtype1_accident \
#     --dataset deepaccident \
#     --target_dir data/deepaccident/processed/ \
#     --split mini \
#     --scene_ids 0 \
#     --workers 8 \
#     --process_keys dynamic_masks

# python datasets/tools/extract_masks.py \
#     --data_root data/deepaccident/processed/mini \
#     --segformer_path=SegFormer \
#     --checkpoint=SegFormer/pretrained/segformer.b5.1024x1024.city.160k.pth \
#     --scene_ids 0 \
#     --process_dynamic_mask

python datasets/tools/humanpose_process.py \
    --dataset deepaccident \
    --data_root data/deepaccident/processed/mini \
    --scene_ids 1 3 4 5 6 \
    --save_temp --verbose