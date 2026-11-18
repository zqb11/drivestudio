#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source /home/zqb/anaconda3/etc/profile.d/conda.sh
conda activate drivestudio

python -u -m tools.eval \
    --resume_from work_dirs/deepaccident_mini/000/omnire+id+residual_clean_loss/checkpoint_final.pth \
    "$@" \
    render.render_full=true \
    render.render_test=false \
    render.render_novel=null
