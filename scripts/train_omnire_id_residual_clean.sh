#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/zqb/anaconda3/etc/profile.d/conda.sh
conda activate drivestudio

python -u -m tools.train \
    --config_file configs/omnire_id_residual_clean.yaml \
    --output_root work_dirs \
    --project deepaccident_mini/002 \
    --run_name omnire+id+residual_clean \
    "$@" \
    data.scene_idx=2 \
    trainer.optim.num_iters=40000
