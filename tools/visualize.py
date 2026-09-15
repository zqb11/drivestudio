from typing import List, Optional
from omegaconf import OmegaConf
import os
import argparse
import time
import json
import wandb
import logging

import torch
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)# 日志级别设置为INFO，这样INFO、WARNING都能显示
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    args.enable_wandb = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)
    
    # 若是da，viser坐标系启动左手系转换
    if dataset.data_cfg['dataset'] == 'deepaccident':
        enable_left_hand_coord = True
    else:
        enable_left_hand_coord = False
        
    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
        #camera_data=dataset.pixel_source.camera_data,
        enable_left_hand_coord=enable_left_hand_coord
    )
    
    # Resume from checkpoint重新加载模型，断点续训
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    logger.info(
        f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
    )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
# 训练后模型高斯场景图可视化
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")   
    
    parser.add_argument("--resume_from", default="work_dirs/omnire/deepaccident_mini_0_6cams_20260622/checkpoint_final.pth", help="path to checkpoint to resume from", type=str)
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=1024, help="viewer port")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)