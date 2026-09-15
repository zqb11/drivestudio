"""
导出场景高斯和天空模型为 PLY 格式的点云文件

支持功能：
- 导出所有高斯节点（Background、RigidNodes、DeformableNodes、SMPLNodes）
- 导出 Sky 模型（球面采样）
- 轻量级启动（无需加载完整数据集）
- 合并导出为单一 PLY 文件

使用示例：
    python tools/export_gaussians.py \
        --checkpoint work_dirs/drivestudio/omnire_23/checkpoint_final.pth \
        --output_dir ./output/
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import numpy as np
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import export_points_to_ply, import_str

logger = logging.getLogger(__name__)


def get_metadata_from_checkpoint(ckpt_path: str) -> int:
    """从检查点的 Affine embedding 推断 num_train_images"""
    try:
        state_dict = torch.load(ckpt_path, map_location='cpu')
        if 'models' in state_dict:
            models_dict = state_dict['models']
            if 'Affine' in models_dict:
                affine_state = models_dict['Affine']
                if 'embedding.weight' in affine_state:
                    num_train_images = affine_state['embedding.weight'].shape[0]
                    logger.info(f"✓ 从检查点推断: num_train_images={num_train_images}")
                    return num_train_images
        logger.warning("⚠ 无法从 Affine 推断 num_train_images，使用默认值 100")
    except Exception as e:
        logger.error(f"❌ 加载检查点失败: {e}")

    return 100


def get_metadata_from_disk(data_cfg) -> int:
    """快速获取 num_timesteps（仅读目录，无需加载完整数据集）"""
    try:
        # 确定数据路径
        try:
            data_path = os.path.join(
                data_cfg.data_root,
                f"{int(data_cfg.scene_idx):03d}"
            )
        except:
            data_path = os.path.join(data_cfg.data_root, data_cfg.scene_idx)

        # 快速获取总帧数
        if os.path.exists(os.path.join(data_path, "ego_pose")):
            total_frames = len(os.listdir(os.path.join(data_path, "ego_pose")))
        elif os.path.exists(os.path.join(data_path, "lidar_pose")):
            total_frames = len(os.listdir(os.path.join(data_path, "lidar_pose")))
        else:
            logger.warning("⚠ 无法确定帧数，使用默认值 100")
            return 100

        # 计算时间步范围
        if data_cfg.end_timestep == -1:
            end_timestep = total_frames - 1
        else:
            end_timestep = data_cfg.end_timestep

        num_timesteps = end_timestep - data_cfg.start_timestep + 1
        logger.info(f"✓ 从目录推断: num_timesteps={num_timesteps}")
        return num_timesteps

    except Exception as e:
        logger.warning(f"⚠ 推断 num_timesteps 失败: {e}，使用默认值 100")
        return 100


def generate_sphere_samples(num_samples: int = 5000) -> torch.Tensor:
    """
    使用 Fibonacci 球面采样生成单位球面上的均匀分布点

    Args:
        num_samples: 采样点数

    Returns:
        points: [num_samples, 3] 单位向量
    """
    # Fibonacci 球面采样
    indices = torch.arange(0, num_samples, dtype=torch.float32)
    theta = 2.0 * np.pi * indices / (1.0 + np.sqrt(5.0)) / 2.0  # golden angle
    phi = torch.acos(1.0 - 2.0 * indices / num_samples)

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    points = torch.stack([x, y, z], dim=-1)  # [num_samples, 3]
    return points


def export_sky_to_ply_data(
    sky_model, num_samples: int = 5000, device: str = 'cuda'
) -> tuple:
    """
    导出 Sky 模型的球面采样数据

    Args:
        sky_model: Sky MLP 模型
        num_samples: 球面采样点数
        device: 计算设备

    Returns:
        (points, colors): 点坐标和颜色
    """
    logger.info(f"🌤️  导出 Sky 球面采样 ({num_samples} 点)...")

    # 生成球面采样点
    sphere_points = generate_sphere_samples(num_samples)
    sphere_points = sphere_points.to(device)

    # 调用 Sky MLP 获得颜色
    sky_model.eval()
    with torch.no_grad():
        image_infos = {"viewdirs": sphere_points}
        rgb_sky = sky_model(image_infos)  # [num_samples, 3]

    # 验证颜色范围
    rgb_sky = torch.clamp(rgb_sky, 0.0, 1.0)

    logger.info(f"✓ Sky 采样完成: {num_samples} 点")
    return sphere_points.cpu(), rgb_sky.cpu()


def main(args):
    """主导出流程"""
    print("\n" + "="*70)
    print("🚀 Gaussian 点云导出工具")
    print("="*70)
    print(f"📦 检查点: {args.checkpoint}")
    print(f"📁 输出目录: {args.output_dir}")
    print("-"*70)

    # 验证文件存在
    if not os.path.exists(args.checkpoint):
        logger.error(f"❌ 检查点文件不存在: {args.checkpoint}")
        return False

    log_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(log_dir, "config.yaml")
    if not os.path.exists(config_path):
        logger.error(f"❌ 配置文件不存在: {config_path}")
        return False

    # 加载配置
    print("⚙️  加载配置...")
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"✓ 配置加载完成，设备: {device}")

    # 推断参数
    print("-"*70)
    print("📥 推断参数...")
    num_train_images = get_metadata_from_checkpoint(args.checkpoint)
    num_timesteps = get_metadata_from_disk(cfg.data)

    # 默认 scene_aabb
    scene_aabb = torch.tensor([[-50.0, -50.0, -5.0], [50.0, 50.0, 5.0]])

    # 初始化模型
    print("-"*70)
    print("🔨 初始化模型...")
    try:
        trainer = import_str(cfg.trainer.type)(
            **cfg.trainer,
            num_timesteps=num_timesteps,
            model_config=cfg.model,
            num_train_images=num_train_images,
            num_full_images=num_train_images,
            test_set_indices=[],
            scene_aabb=scene_aabb,
            device=device
        )
        logger.info(f"✓ 训练器初始化成功")
    except Exception as e:
        logger.error(f"❌ 训练器初始化失败: {e}")
        return False

    # 加载检查点
    print("-"*70)
    print("💾 加载检查点...")
    try:
        trainer.resume_from_checkpoint(
            ckpt_path=args.checkpoint,
            load_only_model=True
        )
        logger.info(f"✓ 检查点加载完成，模型步数: {trainer.step}")
    except Exception as e:
        logger.warning(f"⚠ 严格加载失败，尝试非严格加载...")
        try:
            state_dict = torch.load(args.checkpoint, map_location=device)
            if 'models' in state_dict:
                for class_name in trainer.models.keys():
                    if class_name in state_dict['models']:
                        trainer.models[class_name].load_state_dict(
                            state_dict['models'][class_name],
                            strict=False
                        )
            if 'step' in state_dict:
                trainer.step = state_dict['step']
            logger.info(f"✓ 非严格加载完成，模型步数: {trainer.step}")
        except Exception as e2:
            logger.error(f"❌ 加载检查点失败: {e2}")
            return False

    # 收集高斯数据
    print("-"*70)
    print("🔍 收集高斯数据...")
    all_means = []
    all_colors = []
    total_points = 0

    for class_name in trainer.gaussian_classes.keys():
        model = trainer.models[class_name]
        model.eval()
        with torch.no_grad():
            means = model._means.cpu()
            colors = model.colors.cpu()  # 已处理的 RGB 值

        num_points = means.shape[0]
        total_points += num_points
        logger.info(f"  ✓ {class_name}: {num_points} 点")

        all_means.append(means)
        all_colors.append(colors)

    # 如果有 Sky 模型，添加球面采样点
    sky_points_count = 0
    if 'Sky' in trainer.models:
        sky_points, sky_colors = export_sky_to_ply_data(
            trainer.models['Sky'],
            num_samples=args.sky_samples,
            device=str(device)
        )
        sky_points_count = sky_points.shape[0]
        total_points += sky_points_count

        all_means.append(sky_points)
        all_colors.append(sky_colors)

    # 合并所有点
    print("-"*70)
    print("🔗 合并数据...")
    combined_means = torch.cat(all_means, dim=0)
    combined_colors = torch.cat(all_colors, dim=0)
    logger.info(f"✓ 合并完成: {total_points} 点总计")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 导出为 PLY
    print("-"*70)
    print("💾 导出 PLY 文件...")
    output_path = os.path.join(args.output_dir, args.output_name)

    try:
        start_time = time.time()
        export_points_to_ply(
            positions=combined_means,
            colors=combined_colors,
            save_path=output_path
        )
        elapsed = time.time() - start_time
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB

        print("\n" + "="*70)
        print("✨ 导出成功！")
        print("="*70)
        print(f"📍 输出文件: {output_path}")
        print(f"📊 统计信息:")
        print(f"   • 总点数: {total_points:,}")
        print(f"   • 高斯节点点数: {total_points - sky_points_count:,}")
        if sky_points_count > 0:
            print(f"   • Sky 采样点数: {sky_points_count:,}")
        print(f"   • 文件大小: {file_size:.2f} MB")
        print(f"   • 导出耗时: {elapsed:.2f}s")
        print("="*70 + "\n")

        return True

    except Exception as e:
        logger.error(f"❌ 导出失败: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    parser = argparse.ArgumentParser(
        description="导出场景高斯和 Sky 模型为 PLY 格式"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/sda1/zqb/PythonProjects/drivestudio/work_dirs/drivestudio/omnire_23/checkpoint_final.pth",
        help="检查点文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/sda1/zqb/PythonProjects/drivestudio/work_dirs/drivestudio/omnire_23/point_cloud/",
        help="输出目录（默认: ./output/）"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="combined_gaussians.ply",
        help="输出文件名（默认: combined_gaussians.ply）"
    )
    parser.add_argument(
        "--sky_samples",
        type=int,
        default=5000,
        help="Sky 球面采样点数（默认: 5000）"
    )

    args = parser.parse_args()

    success = main(args)
    sys.exit(0 if success else 1)
