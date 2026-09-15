from typing import Dict
import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere

logger = logging.getLogger()

class MultiTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)# 初始化高斯场景图、超参、场景规模、损失、指标等
        self.render_each_class = True
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
    # 初始化高斯场景图的高斯和模块，并归一化时间步
    def _init_models(self):
        # gaussian model classes高斯场景图高斯类赋值为枚举整型
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
           
        for class_name, model_cfg in self.model_config.items():# 遍历高斯和模块
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)# 从总配置文件弹出该节点类原始子配置
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)# 更新节点类配置，并重新push进总配置
            # 初始化高斯，包括背景、刚性、形变、smpl
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
            # 初始化模块，包括天空、仿射变换、相机位姿优化
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps注册归一化时间戳
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )# 从刚性节点/可形变节点/smpl节点的动态实例字典初始化刚性实例/可形变实例/smpl实例高斯参数
            return False
        else:
            return True
    # 从加载的数据初始化背景高斯、刚性实例高斯、smpl实例高斯、可形变实例高斯
    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # get instance points获得刚性节点、可形变节点、smpl节点的动态实例字典
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            ) # 获得刚性节点的动态实例字典：实例节点类型、所有时间帧实例坐标系下实例点云坐标、所有时间帧实例框内部点云颜色、所有时间帧实例位姿、实例长宽高、实例在所有时间帧是否存在

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )# 获得可形变节点的动态实例字典：键同上

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )# 获得smpl节点的动态实例字典：实例节点类型、实例姿态、实例位置、实例体型参数、实例长宽高、实例所在时间帧、实例点云坐标、实例点云颜色
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict}# 刚性节点、可形变节点、smpl节点的动态实例字典

        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = []# 无点云的高斯类

        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]

            empty = False
            if class_name == 'Background':
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )# 从雷达数据源中采样雷达点，得到采样点坐标、采样点颜色
                    #logger.info(f"[Background] Sampled {sampled_pts.shape[0]} points from LiDAR")
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                # 额外采样一些点
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere场景球内部随机均匀采样近景点，补盲雷达扫不到的近景区域
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)场景球外部逆距离随机均匀采样远景点，填充无限远景
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))

                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0)
                    random_pts = random_pts * self.scene_radius + self.scene_origin# 采样点缩放平移到新世界坐标系
                    visible_mask = dataset.check_pts_visibility(random_pts)# 过滤额外采样的近景点和远景点，得到可见掩码
                    valid_pts = random_pts[visible_mask]# 相机可见的近景和远景点
                    #logger.info(f"[Background] {valid_pts.shape[0]} random points passed visibility check (from {random_pts.shape[0]} total)")

                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)# 采样点坐标：雷达数据源采样点、额外采样的近景和远景点
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)# 采样点颜色

                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )# 过滤实例框内部的采样点，得到背景采样点坐标、颜色

                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"]
                )# 从背景采样点初始化背景高斯(不透明度是0.1)

            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )# 从刚性节点的动态实例字典初始化刚性实例高斯参数

            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )# 从可形变节点的动态实例字典初始化可形变实例高斯参数

            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )# 从smpl节点的动态实例字典初始化smpl实例高斯参数

            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")

        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")

        logger.info(f"Initialized gaussians from pcd")# pcd-point cloud data
    # 模型基于当前图像帧的图像和相机信息前向预测高斯rgb图、高斯深度图、高斯不透明度、天空rgb图、经过高斯不透明度过滤的天空rgb图、经过外观仿射变换的完整rgb图
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

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]# 当前图像帧的归一化时间戳
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )# 根据当前图像帧时间戳与目标时间戳的距离，选取距离最小目标时间戳的时间步作为当前图像帧时间步 self.normalized_timestamps：注册的归一化时间戳，0-1之间等间隔取100个时间步
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )# 训练过程微调相机位姿，获得图像帧对应的相机信息(图像宽高、内参、初始相机位姿、微调之后的相机位姿)
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )# 获取图像帧相机位姿下的所有高斯属性(冻结可选)
        # 调试：最小最大不透明度
        # print(f"Min/Max opacities: {gs.opacities.squeeze().min()}, {gs.opacities.squeeze().max()}")
        # render gaussians
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)# splats半径的裁剪上限，如果训练器渲染配置中无这个参数，取0
        )# 基于初始化的3D高斯，利用高斯溅射和光栅化渲染，得到结果字典(rgb图、期望深度图、不透明度图)、高斯溅射+光栅化渲染函数
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)# 天空rgb图
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])# 通过高斯rgb图不透明度过滤的天空rgb图：有高斯不显示天空，没有高斯显示天空
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )# 对高斯和天空的混合rgb图做外观仿射变换，得到经过外观仿射变换的完整rgb图
        
        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                    outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
        
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict