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
    original_frames = per_cam_poses[list(per_cam_poses.keys())[0]].shape[0]# 原始前视相机轨迹的时间步数
    """
    问题: DeepAccident 使用左手系(improper rotation, det=-1),
    而 Slerp 只在 SO(3) 空间(proper rotation, det=+1)上工作。

    解决方案: 逆变换 -> 插值 -> 重新组合
    1. 逆变换: 将 DeepAccident 的相机轨迹从opencv坐标系变换到数据集坐标系
    2. 插值: 在纯旋转空间中进行 Slerp
    3. 变换: 重新应用坐标系变换, 将相机轨迹从数据集坐标系变换回opencv坐标系 
    """
    # ========== 【新增】获取坐标系转换矩阵 ==========
    da_opencv2dataset = None
    if dataset_type == "deepaccident":
        from datasets.deepaccident.deepaccident_sourceloader import OPENCV2DATASET as DA_OPENCV2DATASET
        da_opencv2dataset = torch.from_numpy(DA_OPENCV2DATASET).float()
        logger.info("Detected DeepAccident dataset, applying coordinate system correction for trajectory interpolation")

    # ========== 【新增】如果有坐标系变换，进行逆变换 ==========
    if da_opencv2dataset is not None:
        da_dataset2opencv = torch.linalg.inv(da_opencv2dataset)
        per_cam_poses_dataset = {}
        for cam_id, poses in per_cam_poses.items():
            per_cam_poses_dataset[cam_id] = poses @ da_dataset2opencv# 将opencv相机轨迹变回da相机轨迹
        logger.info(f"Applied inverse coordinate transformation to camera poses for {dataset_type}")
    else:
        per_cam_poses_dataset = per_cam_poses

    trajectory_generators = {
        "front_center_interp": front_center_interp,# 前视内插：在前视相机轨迹上选取关键帧位姿，在关键帧位姿之间插入新的时间帧位姿，共同组成新轨迹
        "s_curve": s_curve,# S形轨迹：在前视、前左、前右相机轨迹上选取关键帧位姿，在关键帧位姿之间插入新的时间帧位姿，共同组成S形轨迹
        "three_key_poses": three_key_poses_trajectory,# 弧形轨迹：选取前视首帧、前视与前左或前右插值、前视结束帧作为关键帧位姿，在关键帧位姿之间插入新的时间帧位姿，共同组成弧形轨迹
        #"lane_change": front_center_interp,# 变道轨迹：复用前视内插作为基准轨迹，之后在opencv约定下施加横向平移
    }# 新相机轨迹生成器字典，键为轨迹类型，值为对应的生成函数

    if traj_type not in trajectory_generators:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

    # ========== 【新增】在数据集相机位姿下进行插值 ==========
    interp_traj_dataset = trajectory_generators[traj_type](dataset_type, per_cam_poses_dataset, original_frames, target_frames)

    # ========== 【新增】如果有坐标系变换，变换回去 ==========
    if da_opencv2dataset is not None:
        interp_traj = interp_traj_dataset @ da_opencv2dataset
        logger.info(f"Applied coordinate transformation back to interpolated trajectory for {dataset_type}")
    else:
        interp_traj = interp_traj_dataset

    # ========== 【新增】变道横向平移 ==========
    # 必须放在坐标系变换回opencv约定之后：
    # 1. 此时traj[:, :3, 0]统一是"图像右"方向，不随数据集改变
    # 2. P @ M (M为纯旋转、平移列为0) 不改变平移列，所以放在变换后不会被破坏
    # if traj_type == "lane_change":
    #     interp_traj = apply_lane_change_offset(interp_traj, **(traj_cfg or {}))
    #     logger.info(f"Applied lane change offset to interpolated trajectory: {traj_cfg or 'default'}")

    return interp_traj
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
# 变道横向偏移：在opencv约定的相机轨迹上施加横向平移，保持原朝向
# def apply_lane_change_offset(
#     traj: torch.Tensor,
#     direction: str = "left",
#     lane_width: float = 3.5,
#     start_ratio: float = 0.0,
#     end_ratio: float = 0.5,
# ) -> torch.Tensor:
#     """
#     Apply a lateral (lane-change) offset to a camera trajectory, keeping the original orientation.

#     The input trajectory must be in the OpenCV camera convention (x right, y down, z front),
#     so that traj[:, :3, 0] is the world-space direction of "image right" and
#     -traj[:, :3, 1] is the world-space direction of "image up".

#     Args:
#         traj (torch.Tensor): Camera-to-world poses of shape (N, 4, 4), OpenCV convention.
#         direction (str): "left" or "right" - which way the ego vehicle changes lane.
#         lane_width (float): Lateral displacement in meters.
#         start_ratio (float): Normalized time at which the lane change begins.
#         end_ratio (float): Normalized time at which the lane change completes.

#     Returns:
#         torch.Tensor: Offset poses of shape (N, 4, 4).
#     """
#     if direction not in ("left", "right"):
#         raise ValueError(f"Unknown lane change direction: {direction}, expected 'left' or 'right'")
#     if not 0.0 <= start_ratio < end_ratio <= 1.0:
#         raise ValueError(
#             f"Invalid lane change ratios: start_ratio={start_ratio}, end_ratio={end_ratio}, "
#             "expected 0.0 <= start_ratio < end_ratio <= 1.0"
#         )

#     num_frames = traj.shape[0]

#     # 世界系下的"图像右"和"图像上"方向
#     right = traj[:, :3, 0]# (N, 3)
#     up = -traj[:, :3, 1]# (N, 3)
#     up = torch.nn.functional.normalize(up, dim=-1)

#     # 去掉横轴的垂直分量再归一化，避免相机有俯仰/滚转时变道过程中被抬高或压低
#     # 注意: 这里只用点乘投影，不用叉乘——DeepAccident的c2w是improper rotation，叉乘会得到反向结果
#     right_h = right - (right * up).sum(dim=-1, keepdim=True) * up
#     right_h = torch.nn.functional.normalize(right_h, dim=-1)# (N, 3) 世界系下的水平横向单位向量

#     # 归一化时间，映射到变道区间[start_ratio, end_ratio]
#     if num_frames > 1:
#         normed_time = torch.linspace(0, 1, num_frames, device=traj.device, dtype=traj.dtype)
#     else:
#         normed_time = torch.ones(1, device=traj.device, dtype=traj.dtype)
#     t = ((normed_time - start_ratio) / (end_ratio - start_ratio)).clamp(0.0, 1.0)

#     # smoothstep插值，保证变道起止处横向速度为0，视觉上不突兀
#     s = t * t * (3.0 - 2.0 * t)# (N,)

#     sign = -1.0 if direction == "left" else 1.0# 向左即沿"图像右"的反方向
#     offset = sign * lane_width * s[:, None] * right_h# (N, 3) 逐时间步的横向位移

#     new_traj = traj.clone()
#     new_traj[:, :3, 3] = new_traj[:, :3, 3] + offset# 只改平移，旋转保持原朝向
#     return new_traj
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