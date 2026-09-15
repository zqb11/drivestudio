from typing import Dict
import logging
import os
import json
import joblib
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf

import torch
from torch import Tensor
import pickle

from pytorch3d.transforms import matrix_to_quaternion
from datasets.base.scene_dataset import ModelType
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource, CameraData

logger = logging.getLogger()

# define each class's node type 实例类与节点类的映射关系：节点类是整型枚举变量(0-刚性 1-SMPL 2-可形变)
OBJECT_CLASS_NODE_MAPPING = {
    "car": ModelType.RigidNodes,
    "van": ModelType.RigidNodes,
    "truck": ModelType.RigidNodes,
    "pedestrian": ModelType.SMPLNodes,
    "cyclist": ModelType.DeformableNodes,
    "motorcycle": ModelType.DeformableNodes
}
SMPLNODE_CLASSES = ["pedestrian"]

# OpenCV to Dataset coordinate transformation
# opencv coordinate system: x right, y down, z front
# deepaccident coordinate system: x front, y right, z up
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)

# DeepAccident Camera List:
#0: CAM_FRONT         
#1: CAM_FRONT_LEFT    
#2: CAM_FRONT_RIGHT  
#3: CAM_BACK         
#4: CAM_BACK_LEFT     
#5: CAM_BACK_RIGHT   
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4, 5]

class DeepAccidentCameraData(CameraData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, ego-to-world matrices, etc.
        """
        # load camera intrinsics
        # 1d Array of [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}].
        # ====!! we did not use distortion parameters for simplicity !!====
        # to be improved!!
        intrinsic = np.loadtxt(
            os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt")
        )
        fx, fy, cx, cy = intrinsic[0][1], intrinsic[1][2], intrinsic[0][0], intrinsic[1][0]
        # scale intrinsics w.r.t. load size 根据load size相对于原始图像尺寸的缩放比例调整内参中的焦距和主点坐标
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1], 
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )
        _intrinsics = np.array([[fx, 0, cx], [0, -fy, cy], [0, 0, 1]])
        # load camera extrinsics
        cam_to_ego = np.loadtxt(
            os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt")
        )
        # because we use opencv coordinate system to generate camera rays,
        # we need a transformation matrix to covnert rays from opencv coordinate
        # system to deepaccident coordinate system.
        # opencv coordinate system: x right, y down, z front
        # deepaccident coordinate system: x front, y right, z up
        cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # compute per-image poses and intrinsics
        cam_to_worlds, ego_to_worlds = [], []# ego2worlds-每一帧车辆坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵；cam2worlds-每一帧相机坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
        intrinsics, distortions = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first ego pose as the origin of the world coordinate system.将第一帧车辆坐标系作为新世界坐标系
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):# 遍历后面每一个时间帧
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )# 当前时间帧车辆坐标系->世界坐标系的变换矩阵
            # compute ego_to_world transformation
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current# 当前时间帧车辆坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
            ego_to_worlds.append(ego_to_world)
            # transformation:
            #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            #distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()# 当前视角所有时间帧内参
        self.distortions = None
        #torch.from_numpy(np.stack(distortions, axis=0)).float()# 当前视角所有时间帧畸变参数
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()# 当前视角所有时间帧相机坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
    # 获取指定视角所有时间帧的cam到新world变换矩阵
    @classmethod 
    def get_camera2worlds(cls, data_path: str, cam_id: str, start_timestep: int, end_timestep: int) -> torch.Tensor:
        """
        Returns camera-to-world matrices for the specified camera and time range.

        Args:
            data_path (str): Path to the dataset.
            cam_id (str): Camera ID.
            start_timestep (int): Start timestep.
            end_timestep (int): End timestep.

        Returns:
            torch.Tensor: Camera-to-world matrices of shape (num_frames, 4, 4).
        """
        # Load camera extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_path, "extrinsics", f"{cam_id}.txt"))
        cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # Load ego poses and compute camera-to-world matrices
        cam_to_worlds = []
        ego_to_world_start = np.loadtxt(os.path.join(data_path, "ego_pose", f"{start_timestep:03d}.txt"))
        
        for t in range(start_timestep, end_timestep):# 遍历时间帧，得到所有时间帧cam到新世界坐标系的变换矩阵
            ego_to_world_current = np.loadtxt(os.path.join(data_path, "ego_pose", f"{t:03d}.txt"))
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)

        return torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

class DeepAccidentPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf, 
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_data()# 继承自父类，加载图像信息、实例信息、每个视角的初始下采样因子
    # 加载所有视角的相机参数、图像、掩码、归一化时间、相机索引、每一帧图像的唯一索引，存入camera_data字典
    def load_cameras(self):
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)# 时间步
        self.register_normalized_timestamps()# 归一化时间戳
        # 遍历所有视角
        for idx, cam_id in enumerate(self.camera_list):
            logger.info(f"Loading camera {cam_id}")
            camera = DeepAccidentCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
            )# 加载当前视角的相机参数、图像、掩码
            camera.load_time(self.normalized_time)# # 加载当前视角归一化时间戳
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx# 当前视角每一帧图像的唯一索引
            camera.set_unique_ids(
                unique_cam_idx = idx,
                unique_img_idx = unique_img_idx
            )# 加载当前视角相机索引、当前视角每一帧图像的唯一索引
            logger.info(f"Camera {camera.cam_name} loaded.")
            self.camera_data[cam_id] = camera# 将当前视角相机信息存入camera_data字典
    # 加载实例相关信息，包括所有时间帧实例坐标系在新世界坐标系位姿、实例平均尺寸、每一帧所包含的有效实例标记、实例id、实例类别对应的节点类型索引，以及行人SMPL参数（如果需要加载）
    def load_objects(self):
        """
        get ground truth bounding boxes of the dynamic objects

        instances_info = {
            "0": # simplified instance id
                {
                    "id": str,
                    "class_name": str,
                    "frame_annotations": {
                        "frame_idx": List,
                        "obj_to_world": List,
                        "box_size": List,
                },
            ...
        }
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """
        instances_info_path = os.path.join(self.data_path, "instances", "instances_info.json")
        frame_instances_path = os.path.join(self.data_path, "instances", "frame_instances.json")
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)
        # get pose of each instance at each frame
        # shape (num_frames, num_instances, 4, 4)
        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))# 所有时间帧实例坐标系在新世界坐标系位姿，初始化为0
        instances_size = np.zeros((num_full_frames, num_instances, 3))# 所有时间帧实例尺寸，初始化为0
        instances_true_id = np.arange(num_instances)# 实例id
        instances_model_types = np.ones(num_instances) * -1# 每个实例类别对应的节点类型索引
        
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )# 以第一帧车辆坐标系作为新世界坐标系，加载第一帧车辆坐标系->世界坐标系的变换矩阵
        for k, v in instances_info.items():# 遍历每一个实例
            instances_model_types[int(k)] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
            for frame_idx, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"], v["frame_annotations"]["obj_to_world"], v["frame_annotations"]["box_size"]):# 遍历实例所在帧、实例坐标系->世界坐标系的变换矩阵、实例尺寸
                # the first ego pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world# 实例坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
                instances_pose[frame_idx, int(k)] = np.array(obj_to_world)
                instances_size[frame_idx, int(k)] = np.array(box_size)
        
        # get frame valid instances
        # shape (num_frames, num_instances)
        per_frame_instance_mask = np.zeros((num_full_frames, num_instances))# 每一帧所包含的有效实例标记
        for frame_idx, valid_instances in frame_instances.items():# 遍历每一帧所包含的实例
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1# 标记每一帧所包含的实例，有该实例为1，无该实例为0
        
        # select the frames that are in the range of start_timestep and end_timestep
        instances_pose = torch.from_numpy(instances_pose[self.start_timestep:self.end_timestep]).float()
        instances_size = torch.from_numpy(instances_size[self.start_timestep:self.end_timestep]).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        instances_model_types = torch.from_numpy(instances_model_types).long()
        per_frame_instance_mask = torch.from_numpy(per_frame_instance_mask[self.start_timestep:self.end_timestep]).bool()
        
        # filter out the instances that are not visible in selected frames排除在时间帧中没有出现过的实例
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)# 实例在所有时间帧出现的次数
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        instances_model_types = instances_model_types[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]
        
        # assign to the class
        # (num_frames, num_instances, 4, 4)
        self.instances_pose = instances_pose# 所有时间帧实例坐标系在新世界坐标系位姿
        # (num_instances, 3) 对每个实例在所有时间帧的尺寸取平均，得到每个实例的尺寸
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(0).unsqueeze(-1)
        # (num_frames, num_instances)
        self.per_frame_instance_mask = per_frame_instance_mask# 所有时间帧所包含的有无(根据label文件)实例掩码，该时间帧有该实例为1，没有为0
        # (num_instances)
        self.instances_true_id = instances_true_id# 实例id
        # (num_instances)
        self.instances_model_types = instances_model_types# 实例类别对应的节点类型索引：0-
        
        if self.data_cfg.load_smpl:
            # Collect camera-to-world matrices for all available cameras
            cam_to_worlds = {}
            for cam_id in AVAILABLE_CAM_LIST:
                cam_to_worlds[cam_id] = DeepAccidentCameraData.get_camera2worlds(
                    self.data_path, 
                    str(cam_id), 
                    self.start_timestep, 
                    self.end_timestep
                )# 所有视角所有时间帧的cam到新世界坐标系的变换矩阵

            # load SMPL parameters 从人类轨迹加载人类smpl字典
            smpl_dict = joblib.load(os.path.join(self.data_path, "humanpose", "smpl.pkl"))
            frame_num = self.end_timestep - self.start_timestep
            
            smpl_human_all = {}# smpl行人字典：key是行人id，value是一个字典，表示当前行人在每个时间帧的参数，包含姿态(全局姿态和局部姿态)、位置(在新世界坐标系的位置)和体型参数 以及 至少一个视角有2D框且有匹配的SMPL模型参数
            for fi in tqdm(range(self.start_timestep, self.end_timestep), desc="Loading SMPL"):# 遍历时间帧
                for instance_id, ins_smpl in smpl_dict.items():# 遍历行人id和smpl参数
                    if instance_id not in smpl_human_all:
                        smpl_human_all[instance_id] = {
                            "smpl_quats": torch.zeros((frame_num, 24, 4), dtype=torch.float32),# smpl姿态参数，控制行人24个骨骼关节旋转，以四元数形式表示
                            "smpl_trans": torch.zeros((frame_num, 3), dtype=torch.float32),
                            "smpl_betas": torch.zeros((frame_num, 10), dtype=torch.float32),# smpl体型参数，控制行人高矮胖瘦
                            "frame_valid": torch.zeros((frame_num), dtype=torch.bool)
                        }# 初始化smpl行人字典参数为0
                        smpl_human_all[instance_id]["smpl_quats"][:, :, 0] = 1.0# 将旋转四元数初始化为[1, 0, 0, 0]，表示没有旋转
                    if ins_smpl["valid_mask"][fi]:# 如果当前时间帧的valid_mask为True，表示该行人在当前时间帧至少一个视角有2D框且有匹配的SMPL模型参数
                        betas = ins_smpl["smpl"]["betas"][fi]
                        smpl_human_all[instance_id]["smpl_betas"][fi - self.start_timestep] = betas# 当前行人在当前时间帧的体型参数
                        
                        body_pose = ins_smpl["smpl"]["body_pose"][fi]# 当前时间帧当前行人的自身局部姿态，由23个骨骼关节的旋转矩阵表示
                        smpl_orient = ins_smpl["smpl"]["global_orient"][fi]# 当前时间帧当前行人在相机坐标系的姿态
                        cam_depend = ins_smpl["selected_cam_idx"][fi].item()# 当前时间帧当前行人smpl模型参数是在哪个相机视角估计出的
                        
                        c2w = cam_to_worlds[cam_depend][fi - self.start_timestep]
                        world_orient = c2w[:3, :3].to(smpl_orient.device) @ smpl_orient.squeeze()# 当前时间帧当前行人在新世界坐标系的姿态
                        smpl_quats = matrix_to_quaternion(
                            torch.cat([world_orient[None, ...], body_pose], dim=0)
                        )# 拼接“当前时间帧当前行人在新世界坐标系的姿态1”与“自身局部姿态23”，并统一转为四元数
                    
                        ii = instances_info[str(instance_id)]['frame_annotations']["frame_idx"].index(fi)
                        o2w = np.array(
                            instances_info[str(instance_id)]['frame_annotations']["obj_to_world"][ii]
                        )
                        o2w = torch.from_numpy(
                            np.linalg.inv(ego_to_world_start) @ o2w
                        )# 当前时间帧当前行人坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
                        # box_size = instances_info[str(instance_id)]['frame_annotations']["box_size"][ii]
                        
                        smpl_human_all[instance_id]["smpl_quats"][fi - self.start_timestep] = smpl_quats# 当前行人在当前时间帧的全局姿态和局部姿态四元数
                        smpl_human_all[instance_id]["smpl_trans"][fi - self.start_timestep] = o2w[:3, 3]# 当前行人在当前时间帧的位于新世界坐标系的位置
                        smpl_human_all[instance_id]["frame_valid"][fi - self.start_timestep] = True# 当前行人在当前时间帧至少一个视角有2D框且有匹配的SMPL模型参数

            self.smpl_human_all = smpl_human_all
            
class DeepAccidentLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        lidar_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(
                os.path.join(self.data_path, "lidar", f"{t:03d}.bin")
            )
        self.lidar_filepaths = np.array(lidar_filepaths)
    # 加载所有时间帧的lidar坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        # Note that in the DeepAccident Dataset, the lidar coordinate system is different
        # from the vehicle coordinate system deepaccident数据集的lidar坐标系与自车坐标系不同
        lidar_to_worlds = []# 所有时间帧的lidar坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            # load lidar_to_ego transformation matrix 
            lidar_to_ego = np.loadtxt(
                os.path.join(self.data_path, "lidar_to_ego", f"{t:03d}.txt")
            )
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            lidar_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current @ lidar_to_ego
            lidar_to_worlds.append(lidar_to_world)

        self.lidar_to_worlds = torch.from_numpy(
            np.stack(lidar_to_worlds, axis=0)
        ).float()# 加载所有时间帧的lidar坐标系->新世界坐标系(首帧车辆坐标系)的变换矩阵
    # 加载所有时间帧新世界坐标系的激光起点、点云单位方向、点云距离、点云可见性掩码、点云颜色、点云归一化时间戳
    def load_lidar(self):
        origins, directions, ranges = [], [], []
        timesteps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True
        ):# 遍历所有时间帧的lidar数据
            lidar_info = np.fromfile(self.lidar_filepaths[t], dtype=np.float32).reshape(-1, 4)
            original_length = len(lidar_info)# 该时间帧原始激光线数/激光点云总数
            accumulated_num_original_rays += original_length# 所有时间帧原始激光点云总数

            lidar_points = torch.from_numpy(lidar_info[:, :3]).float()# 前三个值是该时间帧点云在激光雷达坐标系的坐标
            lidar_origins = torch.zeros_like(lidar_points)# 该时间帧激光起点与点云坐标形状一样，设为全0

            lidar_origins = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T# 新世界坐标系的该时间帧激光起点
            lidar_points = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T# 新世界坐标系的该时间帧点云坐标
            lidar_directions = lidar_points - lidar_origins# 该时间帧点云方向
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)# 该时间帧点云距离
            lidar_directions = lidar_directions / lidar_ranges# 该时间帧点云方向的单位向量
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t# 该时间帧点云时间戳
            accumulated_num_rays += len(lidar_ranges)# 所有时间帧激光点云总数

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            timesteps.append(lidar_timestamp)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )
        # 加载所有时间帧的激光起点、点云单位方向、点云距离、点云可见性掩码、点云颜色、点云归一化时间戳
        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()# 全局可见性掩码，初始化为False，后续投影过程中记录哪些激光点云能投影到相机视野中
        self.colors = torch.ones_like(self.directions)# 激光点云颜色，初始化为白色(1, 1, 1)，用于兼容某些需要 RGB 颜色输入的神经渲染管线

        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()
    # 获取指定时间帧新世界坐标系的激光点云，包括激光起点、点云单位方向、点云距离、点云归一化时间戳和当前帧点云掩码
    def get_lidar_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_time = self.normalized_time[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_time": normalized_time,
            "lidar_mask": self.timesteps == time_idx,
        }
    # 清除所有时间帧的无效点云
    def delete_invisible_pts(self) -> None:
        """
        Clear the unvisible points.
        """
        if self.visible_masks is not None:
            num_bf = self.origins.shape[0]# 所有时间帧点云总数
            self.origins = self.origins[self.visible_masks]
            self.directions = self.directions[self.visible_masks]
            self.ranges = self.ranges[self.visible_masks]
            self._timesteps = self._timesteps[self.visible_masks]
            self._normalized_time = self._normalized_time[self.visible_masks]
            self.colors = self.colors[self.visible_masks]
            logger.info(
                f"[Lidar] {num_bf - self.visible_masks.sum()} out of {num_bf} points are cleared. {self.visible_masks.sum()} points left."
            )
            self.visible_masks = None
        else:
            logger.info("[Lidar] No unvisible points to clear.")