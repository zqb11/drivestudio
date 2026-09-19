# Camera pose manipulation and trajectory generation.
import os
import torch
import numpy as np
from typing import Dict, Optional
import logging

from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger()
# 在关键帧位姿之间插值组成平滑轨迹
def interpolate_poses(key_poses: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Interpolate between key poses to generate a smooth trajectory.
    
    Args:
        key_poses (torch.Tensor): Tensor of shape (N, 4, 4) containing key camera poses.
        target_frames (int): Number of frames to interpolate.
    
    Returns:
        torch.Tensor: Interpolated poses of shape (target_frames, 4, 4).
    """
    device = key_poses.device
    key_poses = key_poses.cpu().numpy()
    
    # Separate translation and rotation
    translations = key_poses[:, :3, 3]
    rotations = key_poses[:, :3, :3]
    
    # Create time array
    times = np.linspace(0, 1, len(key_poses))# 关键帧的归一化时间点
    target_times = np.linspace(0, 1, target_frames)# 目标帧的归一化时间点
    
    # Interpolate translations(将关键帧的相机位置一维线性插值成目标帧的相机位置)
    interp_translations = np.stack([
        np.interp(target_times, times, translations[:, i])
        for i in range(3)
    ], axis=-1)
    
    # Interpolate rotations using Slerp(基于关键帧的相机姿态球面线性插值Slerp成目标帧的相机姿态)
    key_rots = R.from_matrix(rotations)
    slerp = Slerp(times, key_rots)
    interp_rotations = slerp(target_times).as_matrix()
    
    # Combine interpolated translations and rotations(将目标帧相机位置和姿态组合成新相机轨迹)
    interp_poses = np.eye(4)[None].repeat(target_frames, axis=0)
    interp_poses[:, :3, :3] = interp_rotations
    interp_poses[:, :3, 3] = interp_translations
    
    return torch.tensor(interp_poses, dtype=torch.float32, device=device)

def look_at_rotation(direction: torch.Tensor, up: torch.Tensor = torch.tensor([0., 0., 1.])) -> torch.Tensor:
    """Calculate rotation matrix to look at a specific direction."""
    front = torch.nn.functional.normalize(direction, dim=-1)
    right = torch.nn.functional.normalize(torch.cross(front, up), dim=-1)
    up = torch.cross(right, front)
    rotation_matrix = torch.stack([right, up, -front], dim=-1)
    return rotation_matrix
# 根据新相机轨迹类型调用生成函数合成新轨迹
def get_interp_novel_trajectories(
    dataset_type: str,
    scene_idx: str,
    per_cam_poses: Dict[int, torch.Tensor],
    traj_type: str = "front_center_interp",
    target_frames: int = 100,
    traj_cfg: Optional[dict] = None
) -> torch.Tensor:
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]# 原始相机轨迹的时间步数
    """
    问题: DeepAccident 使用左手系(improper rotation, det=-1),
    而 Slerp 只在 SO(3) 空间(proper rotation, det=+1)上工作。

    解决方案: 变换 -> 插值 -> 重新组合
    1. 变换: 将相机轨迹从opencv坐标系变换到数据集坐标系
    2. 插值: 在纯旋转空间中进行 Slerp
    3. 变换: 重新应用坐标系变换, 将相机轨迹从数据集坐标系变换回opencv坐标系 
    """
    # ========== 【新增】获取opencv相机->数据集相机的转换矩阵 ==========
    da_opencv2dataset = None
    if dataset_type == "deepaccident":
        from datasets.deepaccident.deepaccident_sourceloader import OPENCV2DATASET as DA_OPENCV2DATASET
        da_opencv2dataset = torch.from_numpy(DA_OPENCV2DATASET).float()
        logger.info("Detected DeepAccident dataset, applying coordinate system correction for trajectory generation")

    # ========== 【新增】如果有坐标系变换，进行轨迹变换 ==========
    if da_opencv2dataset is not None:
        da_dataset2opencv = torch.linalg.inv(da_opencv2dataset)
        per_cam_poses_dataset = {}
        for cam_id, poses in per_cam_poses.items():
            per_cam_poses_dataset[cam_id] = poses @ da_dataset2opencv.to(poses)# 从右往左看，将opencv相机轨迹转换成da相机轨迹
        logger.info(f"Applied inverse coordinate transformation to camera poses for {dataset_type}")
    else:
        per_cam_poses_dataset = per_cam_poses

    trajectory_generators = {
        "front_center_left_2m": front_center_left_2m,
        "front_center_left_3m": front_center_left_3m,
        "front_center_interp": front_center_interp,# 前视内插：在前视相机轨迹上选取关键帧位姿，在关键帧位姿之间插入新的时间帧位姿，共同组成新轨迹
        "s_curve": s_curve,# S形轨迹：在前视、前左、前右相机轨迹上选取关键帧位姿，在关键帧位姿之间插入新的时间帧位姿，共同组成S形轨迹
        "three_key_poses": three_key_poses_trajectory,# 弧形轨迹：选取前视首帧、前视与前左或前右插值、前视结束帧作为关键帧位姿，在关键帧位姿之间插入新的时间帧位姿，共同组成弧形轨迹
    }# 新相机轨迹生成器字典，键为轨迹类型，值为对应的生成函数

    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    # 在数据集相机坐标约定下调用生成器，得到新轨迹
    traj_dataset = trajectory_generators[traj_type](dataset_type, per_cam_poses_dataset, original_frames, target_frames)

    # ========== 【新增】如果有坐标系变换，变换回去 ==========
    if da_opencv2dataset is not None:
        traj = traj_dataset @ da_opencv2dataset.to(traj_dataset)# 将da相机新轨迹变回opencv相机新轨迹
        logger.info(f"Applied coordinate transformation back to generated trajectory for {dataset_type}")
    else:
        traj = traj_dataset

    return traj

def front_center_left_2m(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """将原始前中相机逐帧向自身左侧平移 2 米，不进行插值。

    输入采用 DeepAccident 相机坐标约定：x 向前、y 向右、z 向上。
    保留原始帧数和朝向。original_frames、target_frames 用于兼容
    统一的生成器接口，不用于重采样。
    """
    if dataset_type != "deepaccident":
        raise ValueError(
            "front_center_left_2m requires DeepAccident camera coordinates"
        )
    if 0 not in per_cam_poses:
        raise ValueError(
            "Front center camera (ID 0) is required for front_center_left_2m"
        )

    base_traj = per_cam_poses[0]
    shifted_traj = base_traj.clone()

    # 相机局部 +Y 是右方向，第二列是该方向在世界坐标系中的表示。
    right_world = base_traj[:, :3, 1]
    shifted_traj[:, :3, 3] -= 2.0 * right_world

    if target_frames != base_traj.shape[0]:
        logger.info(
            "front_center_left_2m preserves %d original frames; "
            "target_frames=%d is not applied",
            base_traj.shape[0],
            target_frames,
        )

    return shifted_traj

def front_center_left_3m(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int,
) -> torch.Tensor:
    """将原始前中相机逐帧向自身左侧平移 3 米，不进行插值。

    输入采用 DeepAccident 相机坐标约定：x 向前、y 向右、z 向上。
    保留原始帧数和朝向。original_frames、target_frames 用于兼容
    统一的生成器接口，不用于重采样。
    """
    if dataset_type != "deepaccident":
        raise ValueError(
            "front_center_left_2m requires DeepAccident camera coordinates"
        )
    if 0 not in per_cam_poses:
        raise ValueError(
            "Front center camera (ID 0) is required for front_center_left_2m"
        )

    base_traj = per_cam_poses[0]
    shifted_traj = base_traj.clone()

    # 相机局部 +Y 是右方向，第二列是该方向在世界坐标系中的表示。
    right_world = base_traj[:, :3, 1]
    shifted_traj[:, :3, 3] -= 3.0 * right_world

    if target_frames != base_traj.shape[0]:
        logger.info(
            "front_center_left_2m preserves %d original frames; "
            "target_frames=%d is not applied",
            base_traj.shape[0],
            target_frames,
        )

    return shifted_traj



# 前视内插
def front_center_interp(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int, num_loops: int = 1
) -> torch.Tensor:
    """Interpolate key frames from the front center camera."""
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required for front_center_interp"
    key_poses = per_cam_poses[0][::original_frames//4]# 从前视训练轨迹中每隔(原始帧数//4)帧取一帧作为关键帧，得到关键帧位姿
    if key_poses[-1] is not per_cam_poses[0][-1]:# 包含最后一帧：如果关键帧位姿不包含最后一帧，将最后一帧的位姿加入关键帧位姿
        key_poses = torch.cat([key_poses, per_cam_poses[0][-1:]], dim=0)
    return interpolate_poses(key_poses, target_frames)

# S型内插
def s_curve(
    dataset_type: str, per_cam_poses: Dict[int, torch.Tensor], original_frames: int, target_frames: int
) -> torch.Tensor:
    """Create an S-shaped trajectory using the front three cameras."""
    assert all(cam in per_cam_poses.keys() for cam in [0, 1, 2]), "Front three cameras (IDs 0, 1, 2) are required for s_curve"
    key_poses = torch.cat([# 关键帧位姿列表
        per_cam_poses[0][0:1],# 前视起始帧位姿
        per_cam_poses[1][original_frames//4:original_frames//4+1],# 前左1/4时间帧位姿
        per_cam_poses[0][original_frames//2:original_frames//2+1],# 前视1/2时间帧位姿
        per_cam_poses[2][3*original_frames//4:3*original_frames//4+1],# 前右3/4时间帧位姿
        per_cam_poses[0][-1:]# 前视结束帧位姿
    ], dim=0)
    return interpolate_poses(key_poses, target_frames)
# 弧形内插
def three_key_poses_trajectory(
    dataset_type: str,
    per_cam_poses: Dict[int, torch.Tensor],
    original_frames: int,
    target_frames: int
) -> torch.Tensor:
    """
    Create a trajectory using three key poses:
    1. First frame of front center camera
    2. Middle frame with interpolated rotation and position from camera 1 or 2
    3. Last frame of front center camera

    The rotation of the middle pose is calculated using Slerp between
    the start frame and the middle frame of camera 1 or 2.

    Args:
        dataset_type (str): Type of the dataset (e.g., "waymo", "pandaset", etc.).
        per_cam_poses (Dict[int, torch.Tensor]): Dictionary of camera poses.
        original_frames (int): Number of original frames.
        target_frames (int): Number of frames in the output trajectory.

    Returns:
        torch.Tensor: Trajectory of shape (target_frames, 4, 4).
    """
    assert 0 in per_cam_poses.keys(), "Front center camera (ID 0) is required"
    assert 1 in per_cam_poses.keys() or 2 in per_cam_poses.keys(), "Either camera 1 or camera 2 is required"

    # First key pose: First frame of front center camera
    start_pose = per_cam_poses[0][0]
    key_poses = [start_pose]

    # Select camera for middle frame
    middle_frame = int(original_frames // 2)
    chosen_cam = np.random.choice([1, 2])
    # Second key pose: Middle frame of camera 1 or 2
    middle_pose = per_cam_poses[chosen_cam][middle_frame]

    # Calculate interpolated rotation for middle pose
    start_rotation = R.from_matrix(start_pose[:3, :3].cpu().numpy())
    middle_rotation = R.from_matrix(middle_pose[:3, :3].cpu().numpy())
    slerp = Slerp([0, 1], R.from_quat([start_rotation.as_quat(), middle_rotation.as_quat()]))
    interpolated_rotation = slerp(0.5).as_matrix()# 前视起点旋转 和 前左或前右中点旋转的球面线性插值，得到第二个关键帧旋转

    # Create middle key pose with interpolated rotation and original translation
    middle_key_pose = torch.eye(4, device=start_pose.device)
    middle_key_pose[:3, :3] = torch.tensor(interpolated_rotation, device=start_pose.device)
    middle_key_pose[:3, 3] = middle_pose[:3, 3]  # Keep the original translation： 保留前左或前右中间帧平移
    key_poses.append(middle_key_pose)# 第二个关键帧位姿：起始与前左或前右中间帧的旋转插值 + 前左或前右中间帧平移

    # Third key pose: Last frame of front center camera
    key_poses.append(per_cam_poses[0][-1])

    # Stack the key poses and interpolate
    key_poses = torch.stack(key_poses)
    return interpolate_poses(key_poses, target_frames)
