# DeepAccident ID training

## Run

```bash
bash scripts/train_omnire_id.sh
```

The script activates the existing `drivestudio` conda environment. It trains
from scratch for exactly 40,000 updates, using `configs/omnire_id.yaml` and no
`--resume_from`. Output:

```text
work_dirs/omnire-id/deepaccident_mini_0_6cams_20261018/
```

The config preserves the reference scene, six cameras, 100 frames and image
loading resolution `[1, 1, 1, 1, 1, 1]`. It enables full-set evaluation, and
disables unrelated novel-view rendering. No environment packages are changed.

## Call order and changes

1. `tools/train.py` loads the config and initializes the dataset and models.
   Training begins at update 1; the CLI checkpoint default is now `None`.
2. `MultiTrainer.init_gaussians_from_dataset` uses the actual initialization
   dictionaries to build a global identity mapping. `models/identity.py` stores
   annotation poses, sizes and visibility as buffers and registers a shared
   linear classifier. `identity_mapping.json` records local class ordering,
   dataset indices and original track IDs. Class 0 is static.
3. `VanillaGaussians` and node subclasses initialize only one new Gaussian
   attribute: `_id_features` of shape `[N, 8]`. Densification, pruning and
   instance editing keep it aligned with positions. The attribute is disabled
   by default (`ctrl.id_embedding_dim=0`) in existing configs.
4. `BasicTrainer.collect_gaussians` attaches features and global instance IDs
   through `dataclass_gs.extras`. A separate differentiable gsplat pass renders
   features, normalized by accumulated alpha. It preserves the RGB render's
   `means2d`/absolute gradient metadata used for densification.
5. `models/2d_id_loss.py` intersects dynamic pixel rays with annotation
   oriented boxes. Unique hits receive an instance label; overlapping,
   unmatched, cross-near-plane, sky and ego pixels are ignored. These are weak
   labels, not ground-truth visible instance segmentations. Static labels are 0.
6. `MultiTrainer.compute_losses` adds normalized pixel CE, weight 0.1, from
   update 1. At **global step >= 25,000**, it also adds spatial identity KL,
   weight 1.5. `models/3d_id_loss.py` searches neighbors in current world space,
   capped by samples/chunks and a 0.5 m radius, using up to 10 neighbors.
   Dynamic anchors need matching structural ID, projected label, probability
   >= 0.8, and opacity > 0.01. Candidates are nearby background Gaussians whose
   projected labels match the anchor's instance. Both KL inputs retain gradient.
7. The classifier is included in the main optimizer. Identity statistics are
   logged separately from the loss dictionary so counts are not backpropagated.
8. Checkpoints are saved at 20,000, 25,000, 30,000 and the final 40,000 updates.
   New ID checkpoints restore the complete identity mapping and classifier.
   Explicit ID resume from old checkpoints without identity metadata raises an
   error instead of silently disabling ID supervision. Existing non-ID configs
   can still load their original checkpoints.

No velocity/lifetime attribute, background identity gate, foreground coverage
loss, node transfer or ID-based layer reassignment is added. Existing motion,
RGB, depth and geometry regularizers remain in use. Identity gradients can
affect geometry and opacity indirectly; the two requested video criteria must
be checked empirically after training.

## Diagnostics and evaluation

Resume full-set evaluation from the final checkpoint without training:

```bash
bash scripts/eval_omnire_id.sh
```

This writes to the run's original `videos/` and `metrics/` directories. Evaluation
renders six cameras at a time and immediately encodes each timestep, avoiding
the full-sequence float image cache. Regional metrics retain their original
per-image averaging and skip empty masks. Progress and cumulative metrics are
flushed to `metrics/frames_full_40000.jsonl`; the final summary JSON is written
after every video writer closes. `--output_postfix` defaults to `_eval` in the
general eval CLI; the script passes an empty postfix explicitly.

- `images/id_labels_00001.png` through `00006.png`: RGB alongside weak labels;
  gray is ignored and black is static.
- `metrics.json`: losses, dynamic identity accuracy, label counts; after step
  25,000 also anchor/candidate/pair counts. Empty sets produce differentiable
  zero losses.
- `videos/full_set_40000_Background_rgbs.mp4` and
  `videos/full_set_40000_Dynamic_rgbs.mp4`: compare against the reference run's
  videos using the same layer composition behavior. Check body, lights, rear,
  occlusion recovery and temporal stability in addition to overall RGB quality.

```bash
source /home/zqb/anaconda3/etc/profile.d/conda.sh
conda activate drivestudio
python -m unittest discover -s tests -v
```
