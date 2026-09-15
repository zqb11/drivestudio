from typing import Dict, Union, Literal
import logging
import os
import cv2
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf

import torch
from torch import Tensor

from models.gaussians.basics import *
from datasets.base.scene_dataset import ModelType
from datasets.base.scene_dataset import SceneDataset
from datasets.base.split_wrapper import SplitWrapper
from utils.visualization import get_layout
from utils.geometry import transform_points
from utils.camera import get_interp_novel_trajectories
from utils.misc import export_points_to_ply, import_str

logger = logging.getLogger()

DEBUG_PCD=False
if DEBUG_PCD:
    DEBUG_OUTPUT_DIR="debug"
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

NAME_TO_NODE = {
    "RigidNodes": ModelType.RigidNodes,
    "SMPLNodes": ModelType.SMPLNodes,
    "DeformableNodes": ModelType.DeformableNodes
}

class DrivingDataset(SceneDataset):
    def __init__(
        self,
        data_cfg: OmegaConf,
    ) -> None:
        super().__init__(data_cfg)
        
        # AVAILABLE DATASETS:
        #   Waymo:    5 Cameras
        #   KITTI:    2 Cameras
        #   NuScenes: 6 Cameras
        #   ArgoVerse:7 Cameras
        #   PandaSet: 6 Cameras
        #   NuPlan:   8 Cameras
        #   DeepAccident: 6 Cameras
        self.type = self.data_cfg.dataset
        try: # For Waymo, NuScenes, ArgoVerse, PandaSet, DeepAccident
            self.data_path = os.path.join(
                self.data_cfg.data_root,
                f"{int(self.scene_idx):03d}"
            )
        except: # For KITTI, NuPlan
            self.data_path = os.path.join(self.data_cfg.data_root, self.scene_idx)
            
        assert os.path.exists(self.data_path), f"{self.data_path} does not exist"
        if os.path.exists(os.path.join(self.data_path, "ego_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "ego_pose")))# 时间帧数
        elif os.path.exists(os.path.join(self.data_path, "lidar_pose")):
            total_frames = len(os.listdir(os.path.join(self.data_path, "lidar_pose")))
        else:
            raise ValueError("Unable to determine the total number of frames. Neither 'ego_pose' nor 'lidar_pose' directories found.")

        # ---- find the number of synchronized frames ---- #
        if self.data_cfg.end_timestep == -1:
            end_timestep = total_frames - 1
        else:
            end_timestep = self.data_cfg.end_timestep
        # to make sure the last timestep is included
        self.end_timestep = end_timestep + 1
        self.start_timestep = self.data_cfg.start_timestep
        
        # ---- create layout for visualization ---- #
        self.layout = get_layout(self.type)# 可视化函数

        # ---- create data source ---- #
        self.pixel_source, self.lidar_source = self.build_data_source()# 创建像素和雷达数据源
        assert self.pixel_source is not None and self.lidar_source is not None, \
            "Must have both pixel source and lidar source"
        self.project_lidar_pts_on_images(
            delete_out_of_view_points=True
        )# 将所有时间帧的雷达点云投影到所有视角所有时间帧图像上，加载每个视角的深度图，清除所有时间帧的无效点云
        self.aabb = self.get_aabb()# 获得贴合场景的aabb最小点和最大点，即场景边界框

        # ---- define train and test indices ---- #
        # note that the timestamps of the pixel source and the lidar source are the same in waymo dataset
        (
            self.train_timesteps,
            self.test_timesteps,
            self.train_indices,
            self.test_indices,
        ) = self.split_train_test()# 训练测试时间步和训练测试图像帧索引

        # ---- create split wrappers ---- #
        image_sets = self.build_split_wrapper()# 创建图像数据集，包含训练图像数据集、测试图像数据集、完整图像数据集
        self.train_image_set, self.test_image_set, self.full_image_set = image_sets
        
        # debug use
        # self.seg_dynamic_instances_in_lidar_frame(-1, frame_idx=0)
        # self.get_init_objects()
        
    @property
    def instance_num(self):
        return len(self.pixel_source.instances_pose[0])
    
    @property
    def frame_num(self):
        return self.pixel_source.num_frames
    
    def get_instance_infos(self):
        return (
            self.pixel_source.instances_pose.clone(),
            self.pixel_source.instances_size.clone(),
            self.pixel_source.instances_model_types.clone(),
            self.pixel_source.per_frame_instance_mask.clone()
        )

    def build_split_wrapper(self):
        train_image_set = SplitWrapper(
            datasource=self.pixel_source,
            # train_indices are img indices, so the length is num_cams * num_train_timesteps
            split_indices=self.train_indices,
            split="train",
        )# 训练图像数据集，包含像素数据源、训练图像帧索引、"train"数据集标记
        full_image_set = SplitWrapper(
            datasource=self.pixel_source,
            # cover all the images
            split_indices=np.arange(self.pixel_source.num_imgs).tolist(),
            split="full",
        )# 完整图像数据集，包含像素数据源、完整图像帧索引、"full"数据集标记
        test_image_set = None
        if len(self.test_indices) > 0:
            test_image_set = SplitWrapper(
                datasource=self.pixel_source,
                # test_indices are img indices, so the length is num_cams * num_test_timesteps
                split_indices=self.test_indices,
                split="test",
            )# 测试图像数据集，包含像素数据源、测试图像帧索引、"test"数据集标记
        image_sets = (train_image_set, test_image_set, full_image_set)
        return image_sets
    # 创建像素和雷达数据源
    def build_data_source(self):
        """
        Create the data source for the dataset.
        """
        # ---- create pixel source ---- #
        pixel_source = import_str(self.data_cfg.pixel_source.type)(
            self.data_cfg.dataset,
            self.data_cfg.pixel_source,
            self.data_path,
            self.start_timestep,
            self.end_timestep,
            device=self.device,
        )# 创建像素数据源，加载图像信息、实例信息、每个视角的初始下采样因子
        pixel_source.to(self.device)
        
        # ---- create lidar source ---- #
        lidar_source = None
        if self.data_cfg.lidar_source.load_lidar:
            lidar_source = import_str(self.data_cfg.lidar_source.type)(
                self.data_cfg.lidar_source,
                self.data_path,
                self.start_timestep,
                self.end_timestep,
                device=self.device,
            )# 创建雷达数据源，加载点云信息和lidar坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
            lidar_source.to(self.device)
            assert (pixel_source._unique_normalized_timestamps - lidar_source._unique_normalized_timestamps).abs().sum().item() == 0., \
                "The timestamps of the pixel source and the lidar source are not synchronized"
        return pixel_source, lidar_source
    # 从雷达数据源中采样雷达点，得到采样点坐标、采样点颜色
    def get_lidar_samples(
        self, 
        num_samples: float = None,
        downsample_factor: float = None,
        return_color=False,
        return_normalized_time=False,
        device: torch.device = torch.device("cpu")
        ) -> Tensor:
        assert self.lidar_source is not None, "Must have lidar source if you want to get init pcd"
        assert (num_samples is None) != (downsample_factor is None), \
            "Must provide either num_samples or downsample_factor, but not both"
        if downsample_factor is not None:
            num_samples = int(len(self.lidar_source.pts_xyz) / downsample_factor)
        if num_samples > len(self.lidar_source.pts_xyz):
            logger.warning(f"num_samples {num_samples} is larger than the number of points {len(self.lidar_source.pts_xyz)}")
            num_samples = len(self.lidar_source.pts_xyz)
        
        # randomly sample points
        sampled_idx = torch.randperm(len(self.lidar_source.pts_xyz))[:num_samples]# 采样索引
        sampled_pts = self.lidar_source.pts_xyz[sampled_idx].to(device)# 采样点坐标
        
        # get color if needed
        sampled_color = None
        if return_color:
            sampled_color = self.lidar_source.colors[sampled_idx].to(device)# 采样点颜色
        
        sampled_time = None
        if return_normalized_time:
            sampled_time = self.lidar_source._normalized_time[sampled_idx].to(device)
            sampled_time = sampled_time[..., None]
        
        return sampled_pts, sampled_color, sampled_time
    
    def seg_dynamic_instances_in_lidar_frame(
        self,
        instance_ids: Union[int, list],
        frame_idx: int
        ):
        if isinstance(instance_ids, int):
            instance_num = len(self.pixel_source.instances_pose[frame_idx])
            assert instance_ids < instance_num, f"instance_id {instance_ids} is larger than the number of instances {instance_num}"
            if instance_ids == -1:
                instance_ids = list(range(instance_num))
            else:
                instance_ids = [instance_ids]
        elif isinstance(instance_ids, list):
            instance_ids = instance_ids
        
        # get the lidar points
        lidar_dict = self.lidar_source.get_lidar_rays(frame_idx)
        lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
        valid_mask = torch.zeros_like(lidar_pts[:, 0]).bool()
        for instance_id in instance_ids:
            is_valid_instance = self.pixel_source.per_frame_instance_mask[frame_idx, instance_id]
            if not is_valid_instance:
                continue
            # get the pose of the instance at the given frame
            o2w = self.pixel_source.instances_pose[frame_idx, instance_id]
            o_size = self.pixel_source.instances_size[instance_id]
            
            # transform the lidar points to the instance's coordinate system
            # instance_pose [4, 4], pts [N, 3]
            w2o = torch.inverse(o2w)
            o_pts = transform_points(lidar_pts, w2o)
            # get the mask of the points that are inside the instance's bounding box
            mask = (
                (o_pts[:, 0] > -o_size[0] / 2)
                & (o_pts[:, 0] < o_size[0] / 2)
                & (o_pts[:, 1] > -o_size[1] / 2)
                & (o_pts[:, 1] < o_size[1] / 2)
                & (o_pts[:, 2] > -o_size[2] / 2)
                & (o_pts[:, 2] < o_size[2] / 2)
            )
            valid_mask = valid_mask | mask

        valid_points = lidar_pts[valid_mask]
        valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][valid_mask]
        
        if DEBUG_PCD:
            export_points_to_ply(
                valid_points,
                valid_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "vehicle_lidar_pts.ply")
            )
            export_points_to_ply(
                lidar_pts,
                self.lidar_source.colors[lidar_dict["lidar_mask"]],
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "lidar_pts.ply")
            )
    # 获得动态实例字典：实例节点类型、所有时间帧实例坐标系下实例点云坐标、所有时间帧实例框内部点云颜色、所有时间帧实例位姿、实例长宽高、实例在所有时间帧是否存在
    def get_init_objects(
        self,
        cur_node_type: Literal["RigidNodes", "DeformableNodes"],
        instance_max_pts: int = 5000,
        only_moving: bool = True,
        traj_length_thres: float = 0.5,
        exclude_smpl: bool = False,
        ):
        """
        return:
            instances_dict: Dict[int, Dict[str, Tensor]]
                keys: instance_id
                values: Dict[str, Tensor]
                    keys: "pts", "colors", "num_pts", "flows"(Optional)
                    values: Tensor

        NOTE: pts are in object coordinate system
        """
        if self.type == "KITTI":
            traj_length_thres = 5.0
            logger.info(f"For KITTI dataset, the trajectory length threshold is set \
                to {traj_length_thres} to filter out noisy short trajectories of static objects")

        instance_dict = {}
        for fi in range(self.frame_num):# 遍历时间帧
            lidar_dict = self.lidar_source.get_lidar_rays(fi)
            lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]# 当前时间帧雷达点云
            for ins_id in range(self.instance_num):# 遍历实例
                instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]# 该时间帧是否有该实例
                o_type = self.pixel_source.instances_model_types[ins_id].item()# 实例类对应的节点类(整型枚举变量：0-刚性 1-SMPL 2-可形变)
                
                if not instance_active:
                    continue
                
                if cur_node_type == "DeformableNodes":# 如果当前处理的是可形变节点
                    if not (
                        o_type == ModelType.DeformableNodes or 
                        o_type == ModelType.SMPLNodes
                    ):# 当前实例对应节点类不是可形变或smpl，跳过该实例
                        continue
                elif cur_node_type == "RigidNodes":# 如果当前处理的是刚性节点
                    if not o_type == ModelType.RigidNodes:# 当前实例对应节点类不是刚性，跳过该实例
                        continue
                
                if exclude_smpl:# 处理节点类是形变节点时，启用“跳过smpl人类实例”
                    # objects with smpl pose will be modeled by SMPLNodes
                    assert cur_node_type == "DeformableNodes", \
                        "Only exclude SMPL for DeformableNodes"
                    true_id = self.pixel_source.instances_true_id[ins_id].item()# 实例id
                    if true_id in self.pixel_source.smpl_human_all.keys():# 如果该实例是smpl人类，跳过
                        continue

                if ins_id not in instance_dict:
                    instance_dict[ins_id] = {
                        "node_type": cur_node_type,
                        "pts": [],
                        "colors": [],
                        # "flows": [],
                    }# 实例字典：实例节点类型、所有时间帧实例坐标系下实例点云坐标、所有时间帧实例点云颜色
                # get the pose of the instance at the given frame
                o2w = self.pixel_source.instances_pose[fi, ins_id]
                o_size = self.pixel_source.instances_size[ins_id]
                # convert the lidar points to the instance's coordinate system
                w2o = torch.inverse(o2w)
                o_pts = transform_points(lidar_pts, w2o)
                # get the mask of the points that are inside the instance's bounding box获取实例框内部点的掩码
                mask = (
                    (o_pts[:, 0] > -o_size[0] / 2)
                    & (o_pts[:, 0] < o_size[0] / 2)
                    & (o_pts[:, 1] > -o_size[1] / 2)
                    & (o_pts[:, 1] < o_size[1] / 2)
                    & (o_pts[:, 2] > -o_size[2] / 2)
                    & (o_pts[:, 2] < o_size[2] / 2)
                )
                valid_pts = o_pts[mask]# 当前时间帧实例内部点云坐标 
                valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][mask]# 当前时间帧实例框内部点云颜色 
                # valid_flows = lidar_dict["lidar_flows"][mask]
                instance_dict[ins_id]["pts"].append(valid_pts)
                instance_dict[ins_id]["colors"].append(valid_colors)
                # instance_dict[ins_id]["flows"].append(valid_flows)
        
        logger.info(f"Aggregating lidar points across {self.frame_num} frames")
        for ins_id in instance_dict:# 遍历实例，得到实例在所有时间帧实例坐标系下点坐标、颜色和个数
            instance_dict[ins_id]["pts"] = torch.cat(instance_dict[ins_id]["pts"], dim=0)# 当前实例在所有时间帧的点坐标
            instance_dict[ins_id]["colors"] = torch.cat(instance_dict[ins_id]["colors"], dim=0)# 当前实例在所有时间帧的点颜色
            # instance_dict[ins_id]["flows"] = torch.cat(instance_dict[ins_id]["flows"], dim=0)
            instance_dict[ins_id]["num_pts"] = instance_dict[ins_id]["pts"].shape[0]# 当前实例在所有时间帧的点个数
            if instance_dict[ins_id]["num_pts"] > instance_max_pts:
                # randomly sample points
                sampled_idx = torch.randperm(instance_dict[ins_id]["num_pts"])[:instance_max_pts]
                instance_dict[ins_id]["pts"] = instance_dict[ins_id]["pts"][sampled_idx]
                instance_dict[ins_id]["colors"] = instance_dict[ins_id]["colors"][sampled_idx]
                # instance_dict[ins_id]["flows"] = instance_dict[ins_id]["flows"][sampled_idx]
                instance_dict[ins_id]["num_pts"] = instance_max_pts
            logger.info(f"Instance {ins_id} has {instance_dict[ins_id]['num_pts']} lidar sample points")
        
        if only_moving:
            # consider only the instances with non-zero flows
            logger.info(f"Filtering out the instances with non-moving trajectories")
            new_instance_dict = {}
            for k, v in instance_dict.items():
                if v["num_pts"] > 0:
                    # flows = v["flows"]
                    # if flows.norm(dim=-1).mean() > moving_thres:
                    #     v.pop("flows")
                    #     new_instance_dict[k] = v
                    #     logger.info(f"Instance {k} has {v['num_pts']} lidar sample points")
                    frame_info = self.pixel_source.per_frame_instance_mask[:, k]# 所有时间帧有无该实例mask
                    instances_pose = self.pixel_source.instances_pose[:, k]# 所有时间帧该实例位姿
                    instances_trans = instances_pose[:, :3, 3]# 所有时间帧该实例平移向量
                    valid_trans = instances_trans[frame_info]# 过滤没有该实例的时间帧
                    traj_length = valid_trans[1:] - valid_trans[:-1]# (1,...,99)-(0,...,98)所有相邻帧的位移向量
                    traj_length = torch.norm(traj_length, dim=-1).sum()# 实例在整个时间帧移动的总距离：所有相邻帧的距离相加
                    if traj_length > traj_length_thres:# 若实例总距离大于总距离阈值，说明是动的，加入动态实例字典
                        new_instance_dict[k] = v
                        logger.info(f"Instance {k} has {v['num_pts']} lidar sample points")
            instance_dict = new_instance_dict# 动态实例字典：实例节点类型、所有时间帧实例框内部点云坐标、所有时间帧实例框内部点云颜色
            
        # get instance info 在动态实例字典添加所有时间帧实例位姿、实例长宽高、实例在所有时间帧是否存在
        for ins_id in instance_dict:
            instance_dict[ins_id]["poses"] = self.pixel_source.instances_pose[:, ins_id]
            instance_dict[ins_id]["size"] = self.pixel_source.instances_size[ins_id]
            instance_dict[ins_id]["frame_info"] = self.pixel_source.per_frame_instance_mask[:, ins_id]
        
        if DEBUG_PCD:
            output_dir = os.path.join(DEBUG_OUTPUT_DIR, "aggregated_instance_lidar_pts")
            os.makedirs(output_dir, exist_ok=True)
            for ins_id in instance_dict:
                export_points_to_ply(
                    instance_dict[ins_id]["pts"],
                    instance_dict[ins_id]["colors"],
                    save_path=os.path.join(output_dir, f"ID={ins_id}.ply")
                )
        return instance_dict
    # 获得smpl节点的动态实例字典：实例节点类型、实例姿态、实例位置、实例体型参数、实例长宽高、实例所在时间帧、实例点云坐标、实例点云颜色
    def get_init_smpl_objects(self, only_moving: bool = False, traj_length_thres: float = 0.5):
        instance_dict = {}
        """
        instance_dict = {
            ins_id: {
                "node_type": str, 
                "pts": Tensor, [frame_num, num_pts, 3]
                "colors": Tensor, [frame_num, num_pts, 3]
                "quats": Tensor, [frame_num, 4]
                "trans": Tensor, [frame_num, 3]
                "size": Tensor, [3]
                "frame_info": Tensor, [frame_num]
        }
        """
        
        for ins_id in range(self.instance_num):# 遍历实例
            true_id = self.pixel_source.instances_true_id[ins_id].item()
            if true_id in self.pixel_source.smpl_human_all.keys():# 过滤不是smpl人类的实例
                if self.pixel_source.smpl_human_all[true_id]["frame_valid"].sum() == 0:
                    continue
                smpl_trans = self.pixel_source.smpl_human_all[true_id]["smpl_trans"]# 当前smpl人类的每个时间帧的新世界坐标系的位置
                frame_info = self.pixel_source.smpl_human_all[true_id]["frame_valid"]# 当前smpl人类在所有时间帧是否可见且SMPL参数是否有效
                if only_moving and traj_length_thres > 0:# 过滤静止smpl人类
                    # compute the distance between two consecutive frames
                    traj_length = smpl_trans[frame_info][1:] - smpl_trans[frame_info][:-1]
                    traj_length = torch.norm(traj_length, dim=-1).sum()
                    if traj_length < traj_length_thres:
                        continue
                smpl_quats = self.pixel_source.smpl_human_all[true_id]["smpl_quats"]# 当前smpl人类在所有时间帧的全局姿态和局部姿态
                smpl_betas = self.pixel_source.smpl_human_all[true_id]["smpl_betas"]# 当前smpl人类在所有时间帧的体型参数
                size = self.pixel_source.instances_size[ins_id]# # 当前smpl人类的长宽高
                # NOTE: set the first frame's betas as the betas of the instance选有效时间帧中第一个作为实例体型参数
                first_frame_betas = smpl_betas[frame_info][0]

                collected_lidar_pts = []
                collected_lidar_colors = []
                for fi in range(self.frame_num):
                    lidar_dict = self.lidar_source.get_lidar_rays(fi)
                    lidar_pts = lidar_dict["lidar_origins"] + lidar_dict["lidar_viewdirs"] * lidar_dict["lidar_ranges"]
                    instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]
                    if not instance_active:
                        continue
                    
                    # get the pose of the instance at the given frame
                    o2w = self.pixel_source.instances_pose[fi, ins_id]
                    o_size = self.pixel_source.instances_size[ins_id]
                    # convert the lidar points to the instance's coordinate system
                    w2o = torch.inverse(o2w)
                    o_pts = transform_points(lidar_pts, w2o)
                    # get the mask of the points that are inside the instance's bounding box
                    mask = (
                        (o_pts[:, 0] > -o_size[0] / 2)
                        & (o_pts[:, 0] < o_size[0] / 2)
                        & (o_pts[:, 1] > -o_size[1] / 2)
                        & (o_pts[:, 1] < o_size[1] / 2)
                        & (o_pts[:, 2] > -o_size[2] / 2)
                        & (o_pts[:, 2] < o_size[2] / 2)
                    )
                    valid_pts = o_pts[mask]
                    valid_colors = self.lidar_source.colors[lidar_dict["lidar_mask"]][mask]
                    # valid_flows = lidar_dict["lidar_flows"][mask]
                    collected_lidar_pts.append(valid_pts)# 所有时间帧实例框内部点云坐标
                    collected_lidar_colors.append(valid_colors)# 所有时间帧实例框内部点云颜色
                
                instance_dict[ins_id] = {
                    "node_type": "SMPLNodes",
                    "smpl_quats": smpl_quats,           # [frame_num, 24, 4]
                    "smpl_trans": smpl_trans,           # [frame_num, 3]
                    "smpl_betas": first_frame_betas,    # [10]
                    "size":       size,                 # [3]
                    "frame_info": frame_info,           # [frame_num]
                    "pts": torch.cat(collected_lidar_pts, dim=0),
                    "colors": torch.cat(collected_lidar_colors, dim=0),
                }# 动态smpl人类实例字典：实例节点类型、实例姿态、实例位置、实例体型参数、实例长宽高、实例所在时间帧、实例点云坐标、实例点云颜色
        
        return instance_dict
    # 过滤实例框内部的采样点，得到背景采样点坐标、颜色
    def filter_pts_in_boxes(
        self,
        seed_pts: Tensor,
        valid_instances_dict: Dict[int, Dict[str, Tensor]],
        seed_colors: Tensor = None,
        seed_time: Tensor = None,
    ):
        """
        This function is used to filter out the points that are inside the bounding boxes of the instances
        """
        if DEBUG_PCD:
            os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)
            export_points_to_ply(
                seed_pts,
                seed_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "original_seed_pts.ply")
            )
        valid_instance_keys = valid_instances_dict.keys()
        
        inside_mask = torch.zeros_like(seed_pts[:, 0]).bool()
        for fi in range(self.frame_num):
            for ins_id in valid_instance_keys:
                instance_active = self.pixel_source.per_frame_instance_mask[fi, ins_id]
                if not instance_active:
                    continue
                # get the pose of the instance at the given frame
                o2w = self.pixel_source.instances_pose[fi, ins_id].to(seed_pts.device)
                o_size = self.pixel_source.instances_size[ins_id].to(seed_pts.device)
                # convert the lidar points to the instance's coordinate system
                w2o = torch.inverse(o2w)
                o_pts = transform_points(seed_pts, w2o)
                # get the mask of the points that are inside the instance's bounding box
                mask = (
                    (o_pts[:, 0] > -o_size[0] / 2)
                    & (o_pts[:, 0] < o_size[0] / 2)
                    & (o_pts[:, 1] > -o_size[1] / 2)
                    & (o_pts[:, 1] < o_size[1] / 2)
                    & (o_pts[:, 2] > -o_size[2] / 2)
                    & (o_pts[:, 2] < o_size[2] / 2)
                )
                inside_mask = inside_mask | mask
        
        # filter out the points that are inside the bounding boxes
        seed_pts = seed_pts[~inside_mask]
        if seed_colors is not None:
            seed_colors = seed_colors[~inside_mask]
        if seed_time is not None:
            seed_time = seed_time[~inside_mask]
        
        if DEBUG_PCD:
            export_points_to_ply(
                seed_pts,
                seed_colors,
                save_path=os.path.join(DEBUG_OUTPUT_DIR, "filtered_seed_pts.ply")
            )
            
            for fi in range(self.frame_num):
                if fi % 10 != 0:
                    continue
                frame_save_dir = os.path.join(DEBUG_OUTPUT_DIR, f"frame_{fi}")
                os.makedirs(frame_save_dir, exist_ok=True)
                for ins_id in valid_instances_dict:
                    # print number of points
                    # print(f"Frame {fi}, Instance {ins_id} has {valid_instances_dict[ins_id]['pts'].shape[0]} points")
                    o2w = self.pixel_source.instances_pose[fi, ins_id]
                    pts_in_obj = valid_instances_dict[ins_id]["pts"]
                    # rotate the points back to the world coordinate system
                    pts_in_world = transform_points(pts_in_obj, o2w)
                    export_points_to_ply(
                        pts_in_world,
                        valid_instances_dict[ins_id]["colors"],
                        save_path=os.path.join(frame_save_dir, f"ID={ins_id}.ply")
                    )
        
        return {
            "pts": seed_pts,
            "colors": seed_colors,
            "time": seed_time
        }
    # 过滤不可见的近景点和远景点，得到可见掩码：排除相机不可见点，即不在宽或高范围内或深度<=0
    def check_pts_visibility(self, pts_xyz):
        # filter out the lidar points that are not visible from the camera
        pts_xyz = pts_xyz.to(self.device)# 新世界坐标系下点云坐标
        valid_mask = torch.zeros_like(pts_xyz[:, 0]).bool()
        # project lidar points to the image plane
        for cam in self.pixel_source.camera_data.values():
            for frame_idx in range(len(cam)):
                intrinsic_4x4 = torch.nn.functional.pad(
                    cam.intrinsics[frame_idx], (0, 1, 0, 1)
                )
                intrinsic_4x4[3, 3] = 1.0
                lidar2img = (
                    intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()
                )# 新世界坐标系->图像
                projected_points = (
                    lidar2img[:3, :3] @ pts_xyz.T + lidar2img[:3, 3:4]
                ).T
                depth = projected_points[:, 2]
                cam_points = projected_points[:, :2] / (depth.unsqueeze(-1) + 1e-6)
                current_valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                )
                valid_mask = valid_mask | current_valid_mask
        return valid_mask
    # 逐场景优化方法按时间步分割训练集和测试集，进而得到训练测试时间步和训练测试图像帧索引
    def split_train_test(self):
        if self.data_cfg.pixel_source.test_image_stride != 0:
            test_timesteps = np.arange(
                # it makes no sense to have test timesteps before the start timestep
                self.data_cfg.pixel_source.test_image_stride,
                self.num_img_timesteps,
                self.data_cfg.pixel_source.test_image_stride,
            )# 测试时间步：按照测试时间步长从所有时间步中等间隔选取测试时间步
        else:# 若测试集的时间步长为0，则不使用测试集，所有时间步都用于训练
            test_timesteps = []
        train_timesteps = np.array(
            [i for i in range(self.num_img_timesteps) if i not in test_timesteps]
        )# 训练时间步：所有时间步中不属于测试时间步的时间步
        logger.info(
            f"Train timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[train_timesteps]}"
        )
        logger.info(
            f"Test timesteps: \n{np.arange(self.start_timestep, self.end_timestep)[test_timesteps]}"
        )

        # propagate the train and test timesteps to the train and test indices
        train_indices, test_indices = [], []
        for t in range(self.num_img_timesteps):# 遍历时间步
            if t in train_timesteps:# 若是训练时间步
                for cam in range(self.pixel_source.num_cams):
                    train_indices.append(t * self.pixel_source.num_cams + cam)# 训练时间步对应的图像帧索引
            elif t in test_timesteps:
                for cam in range(self.pixel_source.num_cams):
                    test_indices.append(t * self.pixel_source.num_cams + cam)
        logger.info(f"Number of train indices: {len(train_indices)}")
        logger.info(f"Train indices: {train_indices}")
        logger.info(f"Number of test indices: {len(test_indices)}")
        logger.info(f"Test indices: {test_indices}")

        # Again, training and testing indices are indices into the full dataset
        # train_indices are img indices, so the length is num_cams * num_train_timesteps
        # but train_timesteps are timesteps, so the length is num_train_timesteps (len(unique_train_timestamps))
        return train_timesteps, test_timesteps, train_indices, test_indices
    # 加载每个视角的目标深度图：将所有时间帧的雷达点云投影到所有视角所有时间帧图像上，清除所有时间帧的无效点云
    def project_lidar_pts_on_images(self, delete_out_of_view_points=True):
        """
        Project the lidar points on the images and attribute the color of the nearest pixel to the lidar point.
        
        Args:
            delete_out_of_view_points: bool
                If True, the lidar points that are not visible from the camera will be removed.
        """
        for cam in self.pixel_source.camera_data.values():# 遍历视角
            lidar_depth_maps = []# 当前视角所有时间帧的深度图
            for frame_idx in tqdm(
                range(len(cam)), 
                desc="Projecting lidar pts on images for camera {}".format(cam.cam_name),
                dynamic_ncols=True
            ):# 遍历时间帧
                normed_time = self.pixel_source.normalized_time[frame_idx]# 获取图像帧的归一化时间戳
                # 获得当前时间帧雷达到新世界转换矩阵
                # lidar_to_world = self.lidar_source.lidar_to_worlds[frame_idx]
                # get lidar depth on image plane
                closest_lidar_idx = self.lidar_source.find_closest_timestep(normed_time)# 找到与图像帧时间戳最接近的雷达帧索引
                lidar_infos = self.lidar_source.get_lidar_rays(closest_lidar_idx)# 获取该时间帧的点云信息，包括激光起点、单位方向向量、距离、归一化时间戳、该时间步的有效点云mask
                lidar_points = (
                    lidar_infos["lidar_origins"]
                    + lidar_infos["lidar_viewdirs"] * lidar_infos["lidar_ranges"]
                )# 计算该时间帧雷达点云的新世界坐标系下三维坐标o+td
                
                # project lidar points to the image plane 
                if cam.undistort:# 图像去相机镜头畸变
                    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                                cam.intrinsics[frame_idx].cpu().numpy(),
                                cam.distortions[frame_idx].cpu().numpy(),
                                (cam.WIDTH, cam.HEIGHT),
                                alpha=1,
                            )# 内参矩阵做相应调整
                    intrinsic_4x4 = torch.nn.functional.pad(
                            torch.from_numpy(new_camera_matrix), (0, 1, 0, 1)
                        ).to(self.device)# 内参矩阵扩展为4x4齐次矩阵：(0, 1, 0, 1)对应(左, 右, 上, 下)，在右边和下边各添加一行和一列，值为0
                else:
                    intrinsic_4x4 = torch.nn.functional.pad(
                        cam.intrinsics[frame_idx], (0, 1, 0, 1)
                    )
                intrinsic_4x4[3, 3] = 1.0# 齐次内参矩阵右下角元素设置为1
                lidar2img = intrinsic_4x4 @ cam.cam_to_worlds[frame_idx].inverse()# 新世界到图像变换矩阵
                lidar_points = (
                    lidar2img[:3, :3] @ lidar_points.T + lidar2img[:3, 3:4]
                ).T # (num_pts, 3) 雷达点在新世界坐标系下
                
                depth = lidar_points[:, 2]# 点云深度
                cam_points = lidar_points[:, :2] / (depth.unsqueeze(-1) + 1e-6) # (num_pts, 2) 点云的像素坐标(浮点数)
                valid_mask = (
                    (cam_points[:, 0] >= 0)
                    & (cam_points[:, 0] < cam.WIDTH)
                    & (cam_points[:, 1] >= 0)
                    & (cam_points[:, 1] < cam.HEIGHT)
                    & (depth > 0)
                ) # (num_pts, ) 有效像素的掩码，满足像素坐标在图像范围内且深度为正
                depth = depth[valid_mask]# 有效像素深度
                _cam_points = cam_points[valid_mask]# 有效像素坐标
                depth_map = torch.zeros(
                    cam.HEIGHT, cam.WIDTH
                ).to(self.device)# 创建一个与图像大小相同的深度图，初始值为0，高表示行数，宽表示列数
                depth_map[
                    _cam_points[:, 1].long(), _cam_points[:, 0].long()
                ] = depth.squeeze(-1)# 深度图：将像素坐标向下取整(存在重合的像素坐标)作为索引，对应的深度值填入深度图中
                # 调试：打印深度图有效像素坐标，发现存在有效像素
                # coords = torch.nonzero(depth_map)
                # print(coords)
                lidar_depth_maps.append(depth_map)
                
                # used to filter out the lidar points that are visible from the camera
                visible_indices = torch.arange(
                    self.lidar_source.num_points, device=self.device
                )[lidar_infos["lidar_mask"]][valid_mask]# 当前视角当前时间帧有效点下标
                
                self.lidar_source.visible_masks[visible_indices] = True# 将当前视角当前时间帧有效点的掩码设置为True
                
                # attribute the color of the nearest pixel to the lidar point
                points_color = cam.images[frame_idx][
                    _cam_points[:, 1].long(), _cam_points[:, 0].long()
                ]# 像素坐标向下取整作为索引，从图像中获取对应像素的颜色值
                self.lidar_source.colors[visible_indices] = points_color# 将颜色值赋给当前视角当前时间帧有效点的颜色属性

            cam.load_depth(
                torch.stack(lidar_depth_maps, dim=0).to(self.device).float()
            )# 加载当前视角所有时间帧的目标深度图
            
        if delete_out_of_view_points:
            self.lidar_source.delete_invisible_pts()# 清除所有时间帧的无效点云
    # 基于配置中的新轨迹类型获得一个或多个新相机轨迹
    def get_novel_render_traj(
        self,
        traj_types: List[str] = ["front_center_interp"],
        target_frames: int = 100,
        traj_cfgs: Optional[Dict[str, dict]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get multiple novel trajectories of the scene for rendering.

        Args:
            traj_types: List[str]
                A list of trajectory types to generate. Options for each type include:
                - "front_center_interp": Interpolate key frames from the front center camera
                - "s_curve": S-shaped trajectory using the front three cameras
                - "three_key_poses": Creates a trajectory using three key poses from different cameras
                - "lane_change": Front center interpolation with a smooth lateral shift into an adjacent lane
            target_frames: int
                The total number of frames for each novel trajectory
            traj_cfgs: Optional[Dict[str, dict]]
                Per-trajectory-type extra keyword arguments, e.g.
                {"lane_change": {"direction": "left", "lane_width": 3.5}}

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are trajectory types and values
            are the generated novel trajectories, each of shape (target_frames, 4, 4)
        """
        per_cam_poses = {}
        for cam_id in self.pixel_source.camera_list:
            per_cam_poses[cam_id] = self.pixel_source.camera_data[cam_id].cam_to_worlds# 原始逐相机轨迹

        traj_cfgs = traj_cfgs or {}
        novel_trajs = {}# 新相机轨迹
        for traj_type in traj_types:
            novel_trajs[traj_type] = get_interp_novel_trajectories(
                self.type,
                self.scene_idx,
                per_cam_poses,
                traj_type,
                target_frames,
                traj_cfgs.get(traj_type)
            )

        return novel_trajs

    def prepare_novel_view_render_data(self, traj: torch.Tensor) -> list:
            """
            Prepare all necessary elements for novel view rendering.

            Args:
                traj (torch.Tensor): Novel view trajectory, shape (N, 4, 4)

            Returns:
                list: List of dicts, each containing elements required for rendering a single frame:
                    - cam_infos: Camera information (extrinsics, intrinsics, image dimensions)
                    - image_infos: Image-related information (indices, normalized time, viewdirs, etc.)
            """
            # Call the PixelSource's method
            return self.pixel_source.prepare_novel_view_render_data(self.type, traj)