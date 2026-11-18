#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/zqb/anaconda3/etc/profile.d/conda.sh
conda activate drivestudio
scene_ids=(004)

for scene_id in "${scene_ids[@]}"; do
    echo "Training DeepAccident mini case ${scene_id}"
    python tools/train.py \
        --config_file configs/omnire_extended_cam_emd.yaml \
        --project deepaccident_mini \
        --run_name "${scene_id}/+emd" \
        --enable_wandb \
        dataset=deepaccident/6cams \
        data.data_root=data/deepaccident/processed/mini \
        "data.scene_idx=$((10#${scene_id}))"
done
