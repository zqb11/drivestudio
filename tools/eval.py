from typing import List, Optional
from contextlib import ExitStack
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
import imageio

import torch
from datasets.base.pixel_source import get_rays
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
)

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)# 日志级别设置为INFO，这样INFO、WARNING都能显示
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


@torch.no_grad()
def render_novel_multicam_videos(
    trainer, dataset, traj, save_path, fps, save_rgb=True, save_layers=True,
):
    """沿新轨迹渲染六相机 RGB 和分层绿底视频，复用数据集排版。"""
    trainer.set_eval()
    pixel_source = dataset.pixel_source
    camera_ids = pixel_source.camera_list
    if 0 not in camera_ids:
        raise ValueError("Multi-camera novel rendering requires front camera (ID 0)")
    device = trainer.device
    front_pose = pixel_source.camera_data[0].cam_to_worlds[0].to(device)
    world_to_front = torch.linalg.inv(front_pose)
    cameras = []
    for cam_id in camera_ids:
        cam = pixel_source.camera_data[cam_id]
        H, W = cam.HEIGHT, cam.WIDTH
        x, y = torch.meshgrid(
            torch.arange(W, device=device), torch.arange(H, device=device),
            indexing="xy",
        )
        cameras.append({
            "cam": cam,
            "relative_pose": world_to_front @ cam.cam_to_worlds[0].to(device),
            "x": x,
            "y": y,
            "pixel_coords": torch.stack([y.float() / H, x.float() / W], dim=-1),
        })

    frame_indices = torch.linspace(0, pixel_source.num_frames - 1, len(traj)).round().long()
    normed_times = torch.linspace(0, 1, len(traj))
    cam_names = [camera["cam"].cam_name for camera in cameras]
    output_paths = {"rgb": save_path} if save_rgb else {}
    if save_layers:
        for layer in ("Background", "Dynamic", "RigidNodes", "DeformableNodes", "SMPLNodes"):
            output_paths[layer] = save_path.removesuffix("rgbs.mp4") + f"{layer}_rgbs.mp4"
            if layer != "Dynamic" and layer not in trainer.gaussian_classes:
                logger.info("No %s nodes; exporting green background for %s", layer, save_path)
    if not output_paths:
        return
    with ExitStack() as stack:
        writers = {
            name: stack.enter_context(imageio.get_writer(
                path, mode="I", fps=fps, ffmpeg_params=["-threads", "2"],
            ))
            for name, path in output_paths.items()
        }
        for i, pose in enumerate(traj):
            frame_idx = frame_indices[i].item()
            frames = {name: [] for name in output_paths}
            for camera in cameras:
                cam = camera["cam"]
                H, W = cam.HEIGHT, cam.WIDTH
                c2w = pose.to(device) @ camera["relative_pose"]
                intrinsics = cam.intrinsics[frame_idx].to(device)
                origins, viewdirs, direction_norm = get_rays(
                    camera["x"].flatten(), camera["y"].flatten(), c2w, intrinsics,
                )
                image_infos = {
                    "origins": origins.reshape(H, W, 3),
                    "viewdirs": viewdirs.reshape(H, W, 3),
                    "direction_norm": direction_norm.reshape(H, W, 1),
                    "pixel_coords": camera["pixel_coords"],
                    "img_idx": cam.unique_img_idx[frame_idx].to(device).expand(H, W),
                    "frame_idx": torch.tensor(frame_idx, device=device),
                    "normed_time": normed_times[i].to(device),
                }
                camera_infos = {
                    "camera_to_world": c2w,
                    "intrinsics": intrinsics,
                    "height": torch.tensor([H], device=device),
                    "width": torch.tensor([W], device=device),
                    "cam_id": torch.tensor(cam.cam_id, device=device),
                    "cam_name": cam.cam_name,
                }
                outputs = trainer(image_infos, camera_infos, novel_view=True)
                green_background = outputs["rgb"].new_tensor([0.0, 177.0, 64.0]) / 255.0
                for name in output_paths:
                    if name == "rgb":
                        rgb = outputs["rgb"].clamp(0, 1)
                    elif f"{name}_rgb" in outputs:
                        opacity = outputs[f"{name}_opacity"]
                        rgb = outputs[f"{name}_rgb"].clamp(0, 1)
                        rgb = rgb + green_background * (1 - opacity)
                    elif name not in trainer.gaussian_classes and name != "Dynamic":
                        rgb = green_background.expand(H, W, 3)
                    else:
                        raise KeyError(f"Missing Gaussian layer output: {name}_rgb")
                    frames[name].append((rgb.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy())
                del outputs
            for name, writer in writers.items():
                tiled_rgb = dataset.layout(frames[name], cam_names)
                writer.append_data(tiled_rgb.astype("uint8"))
            logger.info("Novel videos %s: frame %d/%d", save_path, i + 1, len(traj))


# 可选：基于测试轨迹和完整轨迹渲染视频、评估指标；基于新轨迹渲染视频
@torch.no_grad()
def do_evaluation(
    step: int = 0,
    cfg: OmegaConf = None,
    trainer: BasicTrainer = None,
    dataset: DrivingDataset = None,
    args: argparse.Namespace = None,
    render_keys: Optional[List[str]] = None,
    post_fix: str = "",
    log_metrics: bool = True
):
    trainer.set_eval()

    logger.info("Evaluating Pixels...")
    if dataset.test_image_set is not None and cfg.render.render_test:# 测试集非空，在测试轨迹渲染视频、评估指标
        logger.info("Evaluating Test Set Pixels...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.test_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/test/{k}"] = v
            if args.enable_wandb:# 测试轨迹指标上传wandb云端
                wandb.log(eval_dict)
            test_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_test_{current_time}.json"
            with open(test_metrics_file, "w") as f:
                json.dump(eval_dict, f)
            logger.info(f"Image evaluation metrics saved to {test_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/test_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/test_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_test_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=2,
            verbose=True,
            save_images=False,
        )
        if args.enable_wandb:
            for k, v in vis_frame_dict.items():# 测试集视频帧上传到wandb云端
                wandb.log({"image_rendering/test/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
        
    if cfg.render.render_full:# 在完整轨迹=训练轨迹上渲染视频、评估指标
        logger.info("Evaluating Full Set...")
        render_results = render_images(
            trainer=trainer,
            dataset=dataset.full_image_set,
            compute_metrics=True,
            compute_error_map=cfg.render.vis_error,
        )
        
        if log_metrics:
            eval_dict = {}
            for k, v in render_results.items():
                if k in [
                    "psnr",
                    "ssim",
                    "lpips",
                    "occupied_psnr",
                    "occupied_ssim",
                    "masked_psnr",
                    "masked_ssim",
                    "human_psnr",
                    "human_ssim",
                    "vehicle_psnr",
                    "vehicle_ssim",
                ]:
                    eval_dict[f"image_metrics/full/{k}"] = v
            if args.enable_wandb:# 完整轨迹指标上传wandb云端
                wandb.log(eval_dict)
            full_metrics_file = f"{cfg.log_dir}/metrics{post_fix}/images_full_{current_time}.json"
            with open(full_metrics_file, "w") as f:
                json.dump(eval_dict, f)# 保存完整轨迹上的评估指标
            logger.info(f"Image evaluation metrics saved to {full_metrics_file}")

        if args.render_video_postfix is None:
            video_output_pth = f"{cfg.log_dir}/videos{post_fix}/full_set_{step}.mp4"
        else:
            video_output_pth = (
                f"{cfg.log_dir}/videos{post_fix}/full_set_{step}_{args.render_video_postfix}.mp4"
            )
        vis_frame_dict = save_videos(
            render_results,
            video_output_pth,
            layout=dataset.layout,
            num_timestamps=dataset.num_img_timesteps,
            keys=render_keys,
            num_cams=dataset.pixel_source.num_cams,
            save_seperate_video=cfg.logging.save_seperate_video,
            fps=cfg.render.fps,
            verbose=True,
        )# 将完整轨迹的渲染图像保存成视频
        if args.enable_wandb:# 上传完整轨迹视频帧到wandb云端
            for k, v in vis_frame_dict.items():
                wandb.log({"image_rendering/full/" + k: wandb.Image(v)})
        del render_results, vis_frame_dict
        torch.cuda.empty_cache()
    # 新相机轨迹视频合成
    render_novel_cfg = cfg.render.get("render_novel", None)
    if render_novel_cfg is not None:
        logger.info("Rendering novel views...")
        render_traj = dataset.get_novel_render_traj(
            traj_types=render_novel_cfg.traj_types,
            target_frames=render_novel_cfg.get("frames", dataset.frame_num),
        )# 基于配置中的新轨迹类型获得一个或多个新相机轨迹
        video_output_dir = f"{cfg.log_dir}/videos{post_fix}/novel_{step}"
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)
        
        for traj_type, traj in render_traj.items():
            trajectory_name = traj_type.removeprefix("front_center_")
            trajectory_dir = os.path.join(video_output_dir, f"{trajectory_name}_rgbs")
            os.makedirs(trajectory_dir, exist_ok=True)
            save_path = os.path.join(trajectory_dir, f"{trajectory_name}_rgbs.mp4")
            save_rgb = render_novel_cfg.get("save_rgb", True)
            save_layers = render_novel_cfg.get("save_layers", True)
            render_each_class = getattr(trainer, "render_each_class", False)
            trainer.render_each_class = save_layers
            try:
                render_novel_multicam_videos(
                    trainer, dataset, traj, save_path,
                    fps=render_novel_cfg.get("fps", cfg.render.fps),
                    save_rgb=save_rgb, save_layers=save_layers,
                )
            finally:
                trainer.render_each_class = render_each_class
            logger.info(f"Saved novel view video for trajectory type: {traj_type} to {save_path}")
            
def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
    cfg.log_dir = log_dir
    args.enable_wandb = False
    for folder in [f"videos{args.output_postfix}", f"metrics{args.output_postfix}"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
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
        enable_left_hand_coord=enable_left_hand_coord
    )
    
    # Resume from checkpoint
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
    
    # define render keys
    render_keys = [
        "Dynamic_rgbs",
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    if args.save_catted_videos:
        cfg.logging.save_seperate_video = False
    
    do_evaluation(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
        post_fix=args.output_postfix
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
# 1.评估：在测试集/完整数据集上渲染图像、评估指标 2.新视角合成
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")    
    # eval
    parser.add_argument("--resume_from", default="front_center_interp,s_curve,three_key_poses", help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--output_postfix", type=str, default="", help="optional suffix for output folders; defaults to videos/ and metrics/")
    parser.add_argument("--save_catted_videos", type=bool, default=False, help="visualize lidar on image")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=1024, help="viewer port")
        
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    main(args)
