# AGENTS.md

## 开发环境
```bash
# 激活环境
conda activate drivestudio
```

## 常用命令
### 训练
```bash
python tools/train.py \
    --config_file configs/omnire_extended_cam.yaml \
    dataset=deepaccident/6cams \
    data.scene_idx=0 \
    data.start_timestep=0 \
    data.end_timestep=-1 \
    --enable_viewer --enable_wandb
```
**数据集格式:** `dataset=<dataset>/<num>cams` (例: `waymo/1cams`, `waymo/5cams`)

### 评估
```bash
python tools/eval.py
```

### 推理
```bash
python tools/visualize.py --enable_viewer
```

## 数据路径
- 处理后数据: `data/<dataset>/processed/`
- 场景列表文件: `data/<dataset>_example_scenes.txt`
- 训练输出: `work_dirs/<project>/<run_name>/`

## 项目入口
- 训练: `tools/train.py`
- 评估: `tools/eval.py`
- 推理: `tools/visualize.py`

## 不能修改的目录
- SMPL模型依赖: `third_party/smplx/`
- 人体处理依赖: `third_party/Humans4D/`
- 打包的依赖库: `package/`

## 关键代码约定
### Scene Index格式
- **Waymo/NuScenes/ArgoVerse/PandaSet/DeepAccident:** 零填充3位数字符串 (如 `"023"`)
- **KITTI/NuPlan:** 实际场景名称字符串

### 配置系统 (OmegaConf)
- CLI覆盖使用 `key=value` 语法
- 嵌套键用点号: `data.scene_idx=023`
- 数据集配置在: `configs/datasets/<dataset>/<num>cams.yaml`

### 时间步范围
- 起始帧: `data.start_timestep=0`
- 结束帧 (-1表示最后一帧): `data.end_timestep=-1`

### 关键模块
- 训练器: `models/trainers/base.py` (BasicTrainer), `scene_graph.py` (MultiTrainer)
- 节点高斯: `models/gaussians/vanilla.py` (背景), `models/nodes/rigid.py` (车辆), `smpl.py` (人体), `deformable.py` (非刚体)
- 数据集: `datasets/driving_dataset.py` (统一接口)
- 数据源: `datasets/<dataset>/<dataset>_sourceloader.py`

## 调试检查
- 显存不足 → 降低batch size

