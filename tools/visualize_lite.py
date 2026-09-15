"""
轻量级可视化脚本 - 不加载完整数据集，加速启动速度
"""
from typing import List, Optional, Tuple
from omegaconf import OmegaConf
import os
import argparse
import time
import logging

import torch
import numpy as np
from utils.misc import import_str
from models.trainers import BasicTrainer

logger = logging.getLogger()

def get_metadata_from_checkpoint(ckpt_path: str) -> int:
    """从检查点中读取 num_train_images"""
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')

        # 尝试从 Affine 模型的 embedding 推断 num_train_images
        if 'models' in state_dict:
            models_dict = state_dict['models']
            if 'Affine' in models_dict:
                affine_state = models_dict['Affine']
                if 'embedding.weight' in affine_state:
                    num_train_images = affine_state['embedding.weight'].shape[0]
                    logger.info(f"✓ 从检查点推断: num_train_images={num_train_images}")
                    return num_train_images

        # 如果推断失败，尝试从其他模型推断
        logger.warning("⚠ 无法从 Affine 推断 num_train_images，尝试其他方法...")

    except Exception as e:
        logger.error(f"❌ 加载检查点失败: {e}")

    # 如果全部失败，使用默认值
    logger.warning("⚠ 使用默认值 num_train_images=100 (如不正确请使用 --use_full_dataset)")
    return 100


def get_metadata_from_disk(data_cfg) -> Tuple[int, List[int], np.ndarray]:
    """
    快速获取数据集元数据，不需要加载完整数据集
    返回: (num_timesteps, test_timesteps, scene_aabb)
    """
    dataset_type = data_cfg.dataset

    try:
        # For Waymo, NuScenes, ArgoVerse, PandaSet
        data_path = os.path.join(
            data_cfg.data_root,
            f"{int(data_cfg.scene_idx):03d}"
        )
    except:
        # For KITTI, NuPlan
        data_path = os.path.join(data_cfg.data_root, data_cfg.scene_idx)

    # 快速获取总帧数
    if os.path.exists(os.path.join(data_path, "ego_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "ego_pose")))
    elif os.path.exists(os.path.join(data_path, "lidar_pose")):
        total_frames = len(os.listdir(os.path.join(data_path, "lidar_pose")))
    else:
        raise ValueError("Unable to find ego_pose or lidar_pose directories")

    # 计算时间步范围
    if data_cfg.end_timestep == -1:
        end_timestep = total_frames - 1
    else:
        end_timestep = data_cfg.end_timestep

    num_timesteps = end_timestep - data_cfg.start_timestep + 1

    # 简化的测试索引（可选）
    test_timesteps = list(range(0, num_timesteps, 5))  # 每5帧取一帧作为测试

    # 从配置或默认值获取scene_aabb
    # 注意：这是一个粗略估计，理想情况下应该从数据集中读取
    scene_aabb = torch.tensor([
        [-50.0, -50.0, -5.0],
        [50.0, 50.0, 5.0]
    ], dtype=torch.float32)

    logger.info(f"快速加载元数据: num_timesteps={num_timesteps}, test_timesteps={len(test_timesteps)}")

    return num_timesteps, test_timesteps, scene_aabb


def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*70)
    print("🚀 轻量级Gaussian Splatting可视化工具")
    print("="*70)
    print(f"📦 检查点: {args.resume_from}")
    print(f"💻 设备: {device}")
    print("-"*70)

    # 从检查点获取 num_train_images
    print("📥 读取检查点元数据...")
    num_train_images = get_metadata_from_checkpoint(args.resume_from)

    # 核心：快速获取元数据（不加载完整数据集）
    if args.use_full_dataset:
        print("📂 加载完整数据集...")
        from datasets.driving_dataset import DrivingDataset
        dataset = DrivingDataset(data_cfg=cfg.data)
        num_timesteps = dataset.num_img_timesteps
        test_timesteps = dataset.test_timesteps
        scene_aabb = dataset.get_aabb().reshape(2, 3)
        print(f"✓ 数据集加载完成")
    else:
        print("⚡ 使用轻量级元数据加载...")
        num_timesteps, test_timesteps, scene_aabb = get_metadata_from_disk(cfg.data)
        scene_aabb = scene_aabb.reshape(2, 3)
        print(f"✓ 元数据加载完成")

    # 创建训练器
    print("-"*70)
    print("🔨 初始化训练器...")
    print(f"   • num_timesteps: {num_timesteps}")
    print(f"   • num_train_images: {num_train_images}")
    print(f"   • scene_aabb: {scene_aabb.shape}")
    try:
        trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=num_timesteps,
            model_config=cfg.model,
            num_train_images=num_train_images,  # 从检查点推断
            num_full_images=num_train_images,   # 与 num_train_images 一致
            test_set_indices=test_timesteps,
            scene_aabb=scene_aabb,
            device=device
        )
        print(f"✓ 训练器初始化成功")
    except Exception as e:
        print(f"❌ 训练器初始化失败: {e}")
        raise

    # 加载检查点
    print("-"*70)
    print("💾 加载检查点...")
    try:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        print(f"✓ 加载完成，模型步数: {trainer.step}")
    except RuntimeError as e:
        # 如果严格加载失败，尝试非严格加载
        print(f"⚠ 严格加载失败，尝试非严格加载...")
        try:
            state_dict = torch.load(args.resume_from, map_location=device)
            if 'models' in state_dict:
                for class_name in trainer.models.keys():
                    if class_name in state_dict['models']:
                        try:
                            trainer.models[class_name].load_state_dict(
                                state_dict['models'][class_name],
                                strict=False
                            )
                            print(f"  ✓ {class_name} 加载成功")
                        except Exception as e2:
                            print(f"  ⚠ {class_name} 加载失败: {e2}")
            if 'step' in state_dict:
                trainer.step = state_dict['step']
            print(f"✓ 非严格加载完成，模型步数: {trainer.step}")
        except Exception as e3:
            print(f"❌ 非严格加载也失败: {e3}")
            raise

    # 初始化viewer
    if args.enable_viewer:
        print("-"*70)
        print("🎬 初始化viewer...")
        trainer.init_viewer(port=args.viewer_port)
        print("\n" + "="*70)
        print("🎉 Viser Viewer 启动成功！")
        print("="*70)
        print(f"🎯 支持节点类型: Background + RigidNodes + DeformableNodes + SMPLNodes")
        print(f"⏱️  总帧数: {num_timesteps}")
        print(f"🖼️  训练图片数: {num_train_images}")
        print(f"🚀 模型步数: {trainer.step}")
        print("-"*70)
        print("  • Ctrl+C: 退出")
        print("="*70 + "\n")
        time.sleep(1000000)
    else:
        logger.info("⏭️  跳过viewer初始化 (使用 --enable_viewer 启用)")


if __name__ == "__main__":
    # 配置日志，减少冗余信息
    logging.basicConfig(
        level=logging.WARNING,  # 只显示 WARNING 及以上
        format='%(message)s'
    )

    parser = argparse.ArgumentParser(
        description="轻量级Gaussian Splatting可视化工具 - 快速启动，无需加载完整数据集"
    )

    parser.add_argument(
        "--resume_from",
        default="work_dirs/omnire/deepaccident_mini_0_6cams_20260622/checkpoint_final.pth",
        help="检查点路径",
        type=str
    )
    parser.add_argument(
        "--enable_viewer",
        action="store_true",
        help="启用viser viewer"
    )
    parser.add_argument(
        "--viewer_port",
        type=int,
        default=1024,
        help="viewer服务端口"
    )
    parser.add_argument(
        "--use_full_dataset",
        action="store_true",
        help="使用完整数据集加载（速度慢但数据完整）"
    )
    parser.add_argument(
        "opts",
        help="命令行配置覆盖",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()
    main(args)
