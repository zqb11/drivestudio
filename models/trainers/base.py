from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn

import kornia
from enum import IntEnum
import viser
import nerfview
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.gaussians.basics import *
# from tools.vis.viewer import DynamicViewer

logger = logging.getLogger()
# 枚举整型的节点类型
class GSModelType(IntEnum):
    Background = 0
    RigidNodes = 1
    SMPLNodes = 2
    DeformableNodes = 3

def lr_scheduler_fn(
    cfg: OmegaConf,
    lr_init: float
):
    if cfg.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = cfg.lr_final

    def func(step):
        step = step - cfg.opt_after
        if step < 0:
            return 0.
        
        if step < cfg.warmup_steps:
            if cfg.ramp == "cosine":
                lr = cfg.lr_pre_warmup + (lr_init - cfg.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / cfg.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    cfg.lr_pre_warmup
                    + (lr_init - cfg.lr_pre_warmup) * step / cfg.warmup_steps
                )
        else:
            t = np.clip(
                (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps), 0, 1
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr  # divided by lr_init because the multiplier is with the initial learning rate

    return func

class BasicTrainer(nn.Module):
    def __init__(
        self,
        type: str = "basic",
        optim: OmegaConf = None,
        losses: OmegaConf = None,
        render: OmegaConf = None,
        res_schedule: OmegaConf = None,
        gaussian_optim_general_cfg: OmegaConf = None,
        gaussian_ctrl_general_cfg: OmegaConf = None,
        model_config: OmegaConf = None,
        num_train_images: int = 0,
        num_full_images: int = 0,
        test_set_indices: List[int] = None,
        scene_aabb: torch.Tensor = None,
        device=None,
        #camera_data=None,
        enable_left_hand_coord: bool = False,
    ):
        super().__init__()
        self._type = type
        self.optim_general = optim
        self.losses_dict = losses# 训练器中的损失超参
        self.render_cfg = render
        self.res_schedule = res_schedule
        self.model_config = model_config# 模型超参
        self.num_iters = self.optim_general.get("num_iters", 30000)
        self.gaussian_optim_general_cfg = gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = gaussian_ctrl_general_cfg
        self.step = 0
        self.device = device
        #self.camera_data = camera_data# 相机信息
        self.enable_left_hand_coord = enable_left_hand_coord  # 左手坐标系转换标志
        
        # dataset infos
        self.num_train_images = num_train_images
        self.num_full_images = num_full_images
        
        # init scene scale
        self._init_scene(scene_aabb=scene_aabb)
        
        # init models
        self.models = {}
        self.misc_classes_keys = [
            'Sky', 'Affine', 'CamPose', 'CamPosePerturb'
        ]# 除高斯之外的模块，包括天空、仿射变换、相机位姿、相机位姿微扰
        self.gaussian_classes = {}
        self._init_models()# 初始化高斯场景图的高斯和模块，并归一化时间步
        self.pts_labels = None # will be overwritten in forward
        self.render_dynamic_mask = False
        
        # init losses fn初始化天空不透明度和深度损失函数
        self._init_losses()
        
        # metrics初始化指标和迭代步数
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.step = 0

        # background color初始化背景颜色，全黑
        self.back_color = torch.zeros(3).to(self.device)
    
        # for evaluation初始化当前时间帧索引和测试集图像索引
        self.cur_frame = torch.tensor(0, device=self.device)
        self.test_set_indices = test_set_indices # will be override
        
        # a simple viewer for background visualization
        self.viewer = None
    
    @property
    def in_test_set(self):
        return self.cur_frame.item() in self.test_set_indices
    
    def set_train(self):
        for model in self.models.values():
            model.train()
        self.train()
    
    def set_eval(self):
        for model in self.models.values():
            model.eval()
        self.eval()
    # 获得训练和验证测试阶段的降采样倍数
    def _get_downscale_factor(self):
        if self.training:# 图像分辨率由粗到精的训练策略：每隔double_steps降采样倍数下降一倍，分辨率提升一倍
            return 2 ** max((self.res_schedule.downscale_times - self.step // self.res_schedule.double_steps), 0)# 降采样倍数
        else:# 原图分辨率
            return 1# 降采样倍数为1
    # 更新节点类的优化和控制配置
    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get('optim', None)
        class_ctrl_cfg = model_cfg.get('ctrl', None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()# 拷贝全局通用优化配置
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()# 拷贝全局通用控制配置
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg['optim'] = new_optim_cfg
        model_cfg['ctrl'] = new_ctrl_cfg

        return model_cfg
    # 初始化场景规模，包括场景原点和半径
    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2# 场景原点：3D包围盒几何中心
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1# 场景半径：3D包围盒最长的那条边/2，得到最长维度的半径，*1.1使得更长一些
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin
        logger.info(f"scene origin: {scene_origin}")
        logger.info(f"scene radius: {scene_radius}")
    
    def _init_models(self) -> None:
        raise NotImplementedError("Please implement the _init_models function")
    
    def initialize_optimizer(self) -> None:
        # get param groups first
        self.param_groups = {}
        for class_name, model in self.models.items():
            self.param_groups.update(model.get_param_groups())
                 
        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            class_optim_cfg = class_cfg["optim"]
            
            raw_optim_cfg = class_optim_cfg.get(component_name, None)
            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
            if raw_optim_cfg.get("lr_final", None) is not None:
                sched_cfg = OmegaConf.create({
                    "opt_after": raw_optim_cfg.get('opt_after', 0),
                    "warmup_steps": raw_optim_cfg.get('warmup_steps', 0),
                    "max_steps": raw_optim_cfg.get('max_steps', self.num_iters),
                    "lr_pre_warmup": raw_optim_cfg.get('lr_pre_warmup', 1.0e-8),
                    "lr_final": raw_optim_cfg.get('lr_final', None),
                    "ramp": raw_optim_cfg.get('ramp', "cosine"),
                })
                # scale the learning rate according to the scene scale
                sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                sched_cfg.lr_final = sched_cfg.lr_final * lr_scale_factor if sched_cfg.lr_final is not None else None
                # adjust max_steps to account for opt_after
                sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
    # 初始化天空不透明度和深度损失函数
    def _init_losses(self) -> None:
        sky_opacity_loss_fn = None
        if "Sky" in self.models:
            if self.losses_dict.mask.opacity_loss_type == "bce":
                from models.losses import binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: binary_cross_entropy(pred, gt, reduction="mean")# 初始化天空不透明度损失函数：输入参数是预测天空不透明度、目标天空掩码，返回值是BCE损失
            elif self.losses_dict.mask.opacity_loss_type == "safe_bce":
                from models.losses import safe_binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: safe_binary_cross_entropy(pred, gt, limit=0.1, reduction="mean")
        self.sky_opacity_loss_fn = sky_opacity_loss_fn
        
        depth_loss_fn = None
        depth_loss_cfg = self.losses_dict.get("depth", None)
        if depth_loss_cfg is not None:
            from models.losses import DepthLoss
            depth_loss_fn = DepthLoss(
                loss_type=depth_loss_cfg.loss_type,
                normalize=depth_loss_cfg.normalize,
                use_inverse_depth=depth_loss_cfg.inverse_depth,
            )# 初始化深度损失函数：l1损失，有效的深度上限，mean_on_hit
        self.depth_loss_fn = depth_loss_fn
    
    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()
    
    def optimizer_step(self) -> None:
        # for params_name, optimizer in self.optimizers.items():
        #     class_name = params_name.split("#")[0]
        #     component_name = params_name.split("#")[1]
        #     max_norm = self.model_config[class_name]["optim"][component_name].get("max_norm", None)
        #     if max_norm is not None:
        #         self.grad_scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(self.param_groups[params_name], max_norm)
        #     if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
        #         self.grad_scaler.step(optimizer)
        self.optimizer.step()

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step
        for class_name in self.gaussian_classes.keys():
            self.models[class_name].preprocess_per_train_step(step)

        # viewer
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()
        
    def postprocess_per_train_step(self, step: int) -> None:
        radii = self.info["radii"]
        if self.render_cfg.absgrad:
            grads = self.info["means2d"].absgrad.clone()# 取绝对梯度
        else:
            grads = self.info["means2d"].grad.clone()
        grads[..., 0] *= self.info["width"] / 2.0 * self.render_cfg.batch_size# 还原尺度
        grads[..., 1] *= self.info["height"] / 2.0 * self.render_cfg.batch_size
        # 按类别切分并分发
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            
            self.models[class_name].postprocess_per_train_step(
                step=step,
                optimizer=self.optimizer,
                radii=radii[0, gaussian_mask],
                xys_grad=grads[0, gaussian_mask],
                last_size=max(self.info["width"], self.info["height"])
            )
        
        # viewer
        if self.viewer is not None:
            num_train_rays_per_step = self.render_cfg.batch_size * self.info["width"] * self.info["height"]
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
            num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)
    # 更新高斯类别的2D溅射块半径，是后续点云密化、修剪或调整学习率的几何依据
    def update_visibility_filter(self) -> None:
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]# 3D高斯球类别掩码，属于这个类别的为true
            self.models[class_name].cur_radii = self.info["radii"][0, gaussian_mask]# 更新该类别的2D溅射块半径
    # 训练过程微调相机位姿，获得相机信息(图像宽高、内参、初始相机位姿、微调之后的相机位姿)
    def process_camera(
        self,
        camera_infos: Dict[str, torch.Tensor],
        image_ids: torch.Tensor,
        novel_view: bool = False
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]# 初始相机位姿
        
        if "CamPosePerturb" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

        if "CamPose" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPose"](camtoworlds, image_ids)#  根据相机位姿矫正参数，在相机坐标系下微调相机位姿
        
        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,# 训练过程中微调之后的相机位姿
            camtoworlds_gt=camtoworlds_gt,# 初始相机位姿
            Ks=camera_infos["intrinsics"],
            H=camera_infos["height"],
            W=camera_infos["width"]
        )
        
        return camera_dict
    # 获取当前图像帧相机位姿下的所有高斯属性(冻结可选)
    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor # leave it here for future use
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "class_labels": [],
        }# 当前相机位姿下所有高斯属性
        for class_name in self.gaussian_classes.keys():# 遍历高斯类
            gs = self.models[class_name].get_gaussians(cam)# 获取当前相机位姿下当前类别高斯属性(中心点坐标、尺度、朝向、颜色和不透明度)
            if gs is None:
                continue
            # 调试：平均不透明度
            # print(f"Mean opacities: {gs['_opacities'].mean()}") 
            # 最小最大不透明度
            # print(f"Min/Max opacities: {gs['_opacities'].min()}, {gs['_opacities'].max()}")
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")# 当前图像帧所有高斯类别
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()

        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras=None        # to save some extra information (TODO) more flexible way
        )# 当前图像帧相机位姿下所有高斯属性：有_是原始属性，无_是根据detach_keys决定的冻结或不冻结属性
        
        return gaussians
    # 基于初始化的高斯，利用高斯溅射和光栅化渲染，得到结果字典(rgb图、预测深度图、不透明度图)、高斯溅射+光栅化渲染函数
    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        # 高斯溅射和光栅化渲染，得到rgb图、预测深度图、不透明度图、元数据
        def render_fn(opaticy_mask=None, return_info=False):
            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(),
                colors=gs.rgbs,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,# 近远剪裁面、渲染模式、半径裁剪上限
            )# 高斯溅射和光栅化渲染，得到颜色图+期望深度图、不透明度图、元数据
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"
            
            assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
            rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None]
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], info
        
        # render rgb and opacity
        rgb, depth, opacity, self.info = render_fn(return_info=True)# 高斯溅射和光栅化渲染，得到rgb图、预测深度图(初始化的高斯z用不透明度阿发合成)、不透明度图、元数据
        
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity
        }
        
        if self.training:
            self.info["means2d"].retain_grad()
        
        return results, render_fn
    # 渲染图像外观仿射变换
    def affine_transformation(
        self,
        rgb_blended: torch.Tensor,
        image_infos: Dict[str, torch.Tensor]
        ):
        if "Affine" in self.models:
            affine_trs = self.models['Affine'](image_infos)# 图像帧对应的外观仿射变换矩阵
            rgb_transformed = (affine_trs[..., :3, :3] @ rgb_blended[..., None] + affine_trs[..., :3, 3:])[..., 0]# 对渲染图像进行外观仿射变换：亮度缩放、色彩空间混合、亮度补偿
            
            return rgb_transformed
        else:       
            return rgb_blended
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
            novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        
        return outputs
    
    def backward(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        # ----------------- backward ----------------
        total_loss = sum(loss for loss in loss_dict.values())
        self.grad_scaler.scale(total_loss).backward()
        self.optimizer_step()
        
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            for group in self.optimizer.param_groups:
                if group["name"] in self.lr_schedulers:
                    new_lr = self.lr_schedulers[group["name"]](self.step)
                    group["lr"] = new_lr
                
    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor] 
    ) -> Dict[str, torch.Tensor]:
        # calculate loss
        loss_dict = {}
        
        if "egocar_masks" in image_infos:
            # in the case of egocar, we need to mask out the egocar region
            valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
            
        gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
        predicted_rgb = outputs["rgb"] * valid_loss_mask[..., None]
        
        gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
        pred_occupied_mask = outputs["opacity"].squeeze() * valid_loss_mask
        
        # rgb loss
        Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
        simloss = 1 - self.ssim(gt_rgb.permute(2, 0, 1)[None, ...], predicted_rgb.permute(2, 0, 1)[None, ...])
        loss_dict.update({
            "rgb_loss": self.losses_dict.rgb.w * Ll1,
            "ssim_loss": self.losses_dict.ssim.w * simloss,
        })
        
        # mask loss
        if self.sky_opacity_loss_fn is not None:
            sky_loss_opacity = self.sky_opacity_loss_fn(pred_occupied_mask, gt_occupied_mask) * self.losses_dict.mask.w
            loss_dict.update({"sky_loss_opacity": sky_loss_opacity})
        
        # depth loss 深度损失
        if self.depth_loss_fn is not None:
            gt_depth = image_infos["lidar_depth_map"]# 目标深度图
            lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask# 有效像素(目标深度>0)掩码：有效像素值为1，否则为0
            pred_depth = outputs["depth"]# 预测深度图
            depth_loss = self.depth_loss_fn(pred_depth, gt_depth, lidar_hit_mask)# 计算损失
            lidar_w_decay = self.losses_dict.depth.get("lidar_w_decay", -1)
            if lidar_w_decay > 0:
                decay_weight = np.exp(-self.step / 8000 * lidar_w_decay)
            else:
                decay_weight = 1
            depth_loss = depth_loss * self.losses_dict.depth.w * decay_weight

            loss_dict.update({"depth_loss": depth_loss})
            
        # ----- reg loss -----不透明度熵正则：让不透明度两极化，要么可见要么不可见，避免出现半透明的高斯球
        opacity_entropy_reg = self.losses_dict.get("opacity_entropy", None)
        if opacity_entropy_reg is not None:
            pred_opacity = torch.clamp(outputs["opacity"].squeeze(), 1e-6, 1 - 1e-6)
            loss_dict.update({
                "opacity_entropy_loss": opacity_entropy_reg.w * (-pred_opacity * torch.log(pred_opacity)).mean()
            })
            
        # from pvg: https://github.com/fudan-zvg/PVG/blob/b4162a9135282e0f3c929054f16be1b3fbacd77a/train.py#L161
        # 逆深度平滑正则：平滑深度同时保留边缘
        inverse_depth_smoothness_reg = self.losses_dict.get("inverse_depth_smoothness", None)
        if inverse_depth_smoothness_reg is not None:
            inverse_depth = 1 / (outputs["depth"] + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
                inverse_depth[None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                image_infos["pixels"][None].permute(0, 3, 1, 2)
            )
            loss_dict.update({
                "inverse_depth_smoothness_loss": inverse_depth_smoothness_reg.w * loss_inv_depth
            })
            
        # affine reg loss仿射变换正则：约束仿射变换矩阵，使得它尽量小
        affine_reg = self.losses_dict.get("affine", None)
        if affine_reg is not None and "Affine" in self.models:
            affine_trs = self.models['Affine']({"img_idx": image_infos["img_idx"].flatten()[0]})
            reg_mat = torch.eye(3, device=self.device)
            reg_shift = torch.zeros(3, device=self.device)
            loss_affine = torch.abs(affine_trs[..., :3, :3] - reg_mat).mean() + torch.abs(affine_trs[..., :3, 3:] - reg_shift).mean()
            loss_dict.update({
                "affine_loss": affine_reg.w * loss_affine
            })

        # dynamic region loss动态区域的额外rgb损失：用于增强动态实例的重建质量
        dynamic_region_weighted_losses = self.losses_dict.get("dynamic_region", None)
        if dynamic_region_weighted_losses is not None:
            weight_factor = dynamic_region_weighted_losses.get("w", 1.0)# 损失权重因子，默认为1.0
            start_from = dynamic_region_weighted_losses.get("start_from", 0)# 启动该损失的起始步数，默认0
            if self.step == start_from:
                self.render_dynamic_mask = True# 渲染单独的动态区域掩码，用于后续的动态区域rgb损失计算
            if self.step > start_from and "Dynamic_opacity" in outputs:
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()# 动态区域覆盖像素
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                
                if dynamic_pred_mask.sum() > 0:
                    Ll1 = torch.abs(gt_rgb[dynamic_pred_mask] - predicted_rgb[dynamic_pred_mask]).mean()# 动态区域计算L1损失
                    loss_dict.update({
                        "vehicle_region_rgb_loss": weight_factor * Ll1,# 权重因子*L1损失
                    })
            
        # compute gaussian reg loss收集每个高斯类自己定义的正则化损失
        for class_name in self.gaussian_classes.keys():
            class_reg_loss = self.models[class_name].compute_reg_loss()
            for k, v in class_reg_loss.items():
                loss_dict[f"{class_name}_{k}"] = v
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        psnr = self.psnr(outputs["rgb"], image_infos["pixels"])
        metric_dict.update({"psnr": psnr})
        return metric_dict
    
    def get_gaussian_count(self):
        num_dict = {}
        for class_name in self.gaussian_classes.keys():
            num_dict[class_name] = self.models[class_name].num_points
        return num_dict
    
    def state_dict(self, only_model: bool = True):
        state_dict = super().state_dict()
        state_dict.update({
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "step": self.step,
        })
        if not only_model:
            state_dict.update({
                "optimizer": {k: v.state_dict() for k, v in self.optimizer.items()},
                # "lr_schedulers": {k: v.state_dict() for k, v in self.lr_schedulers.items()},
                # "grad_scaler": self.grad_scaler.state_dict(),
            })
        return state_dict

    def load_state_dict(self, state_dict: dict, load_only_model: bool =True, strict: bool = True):
        step = state_dict.pop("step")
        self.step = step
        logger.info(f"Loading checkpoint at step {step}")

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)
        
        # load model 
        model_state_dict = state_dict.pop("models")
        for class_name in self.models.keys():
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                logger.warning(f"Cannot find {class_name} in the checkpoint")
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
            logger.info(f"{class_name}: {msg}")
        msg = super().load_state_dict(state_dict, strict)
        logger.info(f"BasicTrainer: {msg}")
    # 断点续训
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path)# 加载模型文件为状态字典 
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=True)# 将状态字典加载到模型models中
        
    def save_checkpoint(
        self,
        log_dir: str,
        save_only_model: bool=True,
        is_final: bool=False
    ) -> None:
        """
        Save model to checkpoint.
        """
        if is_final:
            ckpt_path = os.path.join(log_dir, f"checkpoint_final.pth")
        else:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{self.step:05d}.pth")
        torch.save(self.state_dict(only_model=save_only_model), ckpt_path)
        logger.info(f"Saved a checkpoint to {ckpt_path}")
        
    def init_viewer(self, port: int = 8080):
        # a simple viewer for visualization
        self.server = viser.ViserServer(port=port, verbose=False)# viser服务
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            num_frames=self.num_timesteps,
            mode="training",
        )# viser客户端：为相机更新和时间步更新事件分别绑定回调函数，当相机或时间步或同时更新时，利用_viewer_render_fn渲染当前相机位姿、内参、图像分辨率和时间步的rgb图像

    @torch.no_grad()
    def _get_viewdirs(self, K: torch.Tensor, c2w: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        计算每个像素的视线方向（从相机指向像素）

        Args:
            K: 相机内参 [3, 3]
            c2w: 相机位姿 [4, 4]
            H: 图像高度
            W: 图像宽度

        Returns:
            viewdirs: [H, W, 3] 视线方向（世界坐标系）
        """
        # 创建像素坐标 [H, W, 2]
        y, x = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )

        # 反投影到相机坐标系
        # 相机坐标系: x_c = (x - cx) / fx, y_c = (y - cy) / fy, z_c = 1
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]

        x_c = (x - cx) / fx
        y_c = (y - cy) / fy
        z_c = torch.ones_like(x)

        # 相机坐标系中的方向向量
        dirs_c = torch.stack([x_c, y_c, z_c], dim=-1)  # [H, W, 3]

        # 变换到世界坐标系
        # viewdirs = dirs_c @ R.T (R 是旋转矩阵)
        R = c2w[:3, :3]  # [3, 3]
        dirs_w = torch.einsum('ij,hwj->hwi', R, dirs_c)  # [H, W, 3]

        # 归一化
        viewdirs = dirs_w / (dirs_w.norm(dim=-1, keepdim=True) + 1e-8)

        return viewdirs
    # 渲染当前相机位姿、内参、图像分辨率和时间步的rgb图像
    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        #print("Rendering trigger!")# 测试有没有进入回调函数
        W, H = img_wh
        t = int(self.viewer._playback_guis[0].value)
        c2w = camera_state.c2w# 相机位姿 
        K = camera_state.get_K(img_wh)# 相机内参，与垂直视场角有关
        c2w = torch.from_numpy(c2w).float().to(self.device)
        # c2w[:3, 3] = self.camera_data[0].cam_to_worlds[t][:3, 3]# 相机位置替换为不同时间步的单视角相机位置
        # 让c2w相机位置和方向随时间运动：保持相对于自车的相对视角
        # cam_to_worlds = self.camera_data[0].cam_to_worlds  # [num_frames, 4, 4]
        # 初始视角相对于真实轨迹的相对变换
        # # T_rel = c2w_init @ inv(cam_to_worlds[0])
        # cam_to_worlds_np = cam_to_worlds.numpy() if isinstance(cam_to_worlds, torch.Tensor) else cam_to_worlds.cpu().numpy()
        # # 计算相对变换（首帧）
        # w2w_rel = c2w @ np.linalg.inv(cam_to_worlds_np[0])
        # # 应用到当前时间步：c2w_t = c2w_rel @ cam_to_worlds[t]
        # c2w = torch.from_numpy(w2w_rel @ cam_to_worlds_np[t]).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        #breakpoint()# 回调函数内部强制断点调试
        
        # 坐标系转换：从Viser右手系转换到左手系
        # 只在高斯点采用左手坐标系时启用此转换
        if self.enable_left_hand_coord:  
            left_hand_transform = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=self.device
            )
            c2w = left_hand_transform @ c2w 
        
        
        cam = dataclass_camera(
            camtoworlds=c2w,# 估计相机位姿=真值，不做相机轨迹优化
            camtoworlds_gt=c2w,# 目标相机位姿
            Ks=K,
            H=H,
            W=W
        )# 相机数据类
        
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
        }# 高斯属性字典
        
        # 为所有动态节点设置当前帧
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(t)
                
        for class_name in self.gaussian_classes.keys():# 遍历背景、刚体节点、SMPL节点和可变形节点
            gs = self.models[class_name].get_gaussians(cam)# 背景节点：获得当前相机位姿下的节点高斯属性；前景节点：获得当前相机位姿当前时间帧的高斯属性(当前时间帧无效点的不透明度设为0，完全透明)
            if gs is None:
                continue

            for k, _ in gs.items():# 将这类节点高斯属性分别追加到五个键值列表中
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():# 遍历五个键值列表，将每个列表中的张量拼接成一个大张量
            gs_dict[k] = torch.cat(v, dim=0)

        gs = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],
            extras=None
        )# 当前相机位姿下按高斯属性分类的所有节点高斯
        
        # 调用光栅化函数，将按高斯属性分类的所有高斯渲染为rgb+深度图像
        render_colors, alphas, _ = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities.squeeze(),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # 相机外参w2c[C, 4, 4]
            Ks=cam.Ks[None, ...],  # 相机内参[C, 3, 3]
            width=cam.W,
            height=cam.H,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            radius_clip=4.0,  # skip GSs that have small image radius (in pixels)
        )
        # return render_colors[0].cpu().numpy()

        # 核心：渲染 Sky（如果存在）
        rgb_gaussian = render_colors[0, ..., :3]  # 取 RGB 通道，丢弃 depth
        opacity = alphas[0].squeeze(-1)  # [H, W]

        if 'Sky' in self.models:
            # 计算视线方向
            viewdirs = self._get_viewdirs(K, c2w, H, W)  # [H, W, 3]

            # 构造 image_infos
            image_infos = {
                "viewdirs": viewdirs,  # [H, W, 3]
                "img_idx": torch.tensor(t, device=self.device)  # 使用帧索引作为图片索引
            }

            # 渲染 Sky
            sky_model = self.models['Sky']
            rgb_sky = sky_model(image_infos)  # [H, W, 3]

            # 混合：gaussian + sky * (1 - opacity)
            rgb_blended = rgb_gaussian + rgb_sky * (1.0 - opacity[..., None])
            rgb_blended = torch.clamp(rgb_blended, 0, 1)

            # 返回混合后的结果
            return rgb_blended.cpu().numpy()
        else:
            # 如果没有 Sky 模型，直接返回高斯渲染结果
            return rgb_gaussian.cpu().numpy()