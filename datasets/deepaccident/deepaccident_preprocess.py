import json
import os
import pickle

import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2

from datasets.tools.multiprocess_utils import track_parallel_progress


class DeepAccidentProcessor(object):
    """Process DeepAccident dataset.

    DeepAccident contains multi-view accident scenarios with:
    - 5 viewpoints (ego_vehicle, ego_vehicle_behind, other_vehicle,
      other_vehicle_behind, infrastructure)
    - 6 cameras per viewpoint (Front, FrontLeft, FrontRight, Back, BackLeft, BackRight)
    - Calibration data in PKL format
    - LiDAR data in NPZ format
    - Object annotations in text format
    """

    def __init__(
        self,
        load_dir,
        save_dir,
        prefix,
        process_keys=[
            "images",
            "lidar",
            "calib",
            "pose",
            "dynamic_masks",
            "objects"
        ],
        process_id_list=None,
        workers=4,
    ):
        self.process_keys = process_keys
        print("will process keys: ", self.process_keys)

        self.process_id_list = process_id_list
        self.load_dir = load_dir
        self.save_dir = f"{save_dir}/{prefix}"
        self.workers = int(workers)

        # Viewpoint configuration
        self.viewpoints = [
            "ego_vehicle",
            "ego_vehicle_behind",
            "other_vehicle",
            "other_vehicle_behind",
            "infrastructure"
        ]
        self.camera_names = [
            "Camera_Front",
            "Camera_FrontLeft",
            "Camera_FrontRight",
            "Camera_Back",
            "Camera_BackLeft",
            "Camera_BackRight"
        ]

        # Build list of scenarios by finding all unique scenario names
        self.scenario_list = self._build_scenario_list()
        self.create_folder()
        

    def _build_scenario_list(self):
        """Build list of all scenarios in the dataset.检索场景名列表"""
        scenarios = set()
        vp_path = os.path.join(self.load_dir, self.viewpoints[0], "Camera_Front")
        # 将场景名转化为场景id
        if os.path.exists(vp_path):
            for scenario_dir in os.listdir(vp_path):
                if os.path.isdir(os.path.join(vp_path, scenario_dir)):
                    scenarios.add(scenario_dir)
            scenarios = sorted(list(scenarios))
            #scenarios_id = [str(i).zfill(3) for i in range(len(sorted(list(scenarios))))]
        return scenarios

    def create_folder(self):
        """Create output directories for all scenarios.为所有场景建立输入数据目录"""
        for scenario in self.scenario_list:
            for key in self.process_keys:
                if key == "images":
                    os.makedirs(f"{self.save_dir}/{scenario}/images", exist_ok=True)
                elif key == "calib":
                    os.makedirs(f"{self.save_dir}/{scenario}/intrinsics", exist_ok=True)
                    os.makedirs(f"{self.save_dir}/{scenario}/extrinsics", exist_ok=True)
                    os.makedirs(f"{self.save_dir}/{scenario}/lidar_to_ego", exist_ok=True)
                elif key == "lidar":
                    os.makedirs(f"{self.save_dir}/{scenario}/lidar", exist_ok=True)
                elif key == "pose":
                    os.makedirs(f"{self.save_dir}/{scenario}/ego_pose", exist_ok=True)
                elif key == "dynamic_masks":
                    os.makedirs(f"{self.save_dir}/{scenario}/dynamic_masks/all", exist_ok=True)
                    os.makedirs(f"{self.save_dir}/{scenario}/dynamic_masks/human", exist_ok=True)
                    os.makedirs(f"{self.save_dir}/{scenario}/dynamic_masks/vehicle", exist_ok=True)
                elif key == "objects":
                    os.makedirs(f"{self.save_dir}/{scenario}/instances", exist_ok=True)

    def convert(self):
        """Convert action."""
        print("Start converting ...")
        if self.process_id_list is None:
            id_list = self.scenario_list
        else:
            # Convert indices/IDs to actual scenario names
            id_list = []
            for item in self.process_id_list:
                item_int = int(item) if not isinstance(item, int) else item
                if item_int < len(self.scenario_list):
                    id_list.append(self.scenario_list[item_int])
                else:
                    # Try as string directly
                    id_list.append(str(item))
        track_parallel_progress(self.convert_one, id_list, self.workers)

        # Rename scenario directories: scenario_name -> 000, 001, 002, ...
        # for idx, scenario_name in enumerate(self.scenario_list):
        #     old_dir = os.path.join(self.save_dir, scenario_name)
        #     new_dir = os.path.join(self.save_dir, str(idx).zfill(3))
        #     if os.path.exists(old_dir) and old_dir != new_dir:
        #         os.rename(old_dir, new_dir)

        print("\nFinished ...")

    def convert_one(self, scenario_id):
        """Convert single scenario.

        Args:
            scenario_id (str): Scenario ID to process
        """

        # Get number of frames from first camera
        first_camera_path = os.path.join(
            self.load_dir, self.viewpoints[0], self.camera_names[0], scenario_id
        )
        if not os.path.exists(first_camera_path):
            print(f"Scenario {scenario_id} not found, skipping...")
            return

        frame_files = [f for f in os.listdir(first_camera_path) if f.endswith('.jpg')]

        # Extract frame indices from filenames
        frame_indices = sorted([
            int(f.split('_')[-1].replace('.jpg', ''))
            for f in frame_files
        ])

        for frame_idx in tqdm(frame_indices, desc=f"Processing {scenario_id}", dynamic_ncols=True):
            if "images" in self.process_keys:
                self.save_images(scenario_id, frame_idx)
            if "calib" in self.process_keys:
                self.save_calib(scenario_id, frame_idx)
            if "lidar" in self.process_keys:
                self.save_lidar(scenario_id, frame_idx)
            if "pose" in self.process_keys:
                self.save_pose(scenario_id, frame_idx)
            if "dynamic_masks" in self.process_keys:
                self.save_dynamic_mask(scenario_id, frame_idx, 'all')
                self.save_dynamic_mask(scenario_id, frame_idx, 'human')
                self.save_dynamic_mask(scenario_id, frame_idx, 'vehicle')

        if "objects" in self.process_keys:
            instances_info, frame_instances = self.save_objects(scenario_id)
            # Save instances info
            instances_info_path = os.path.join(
                self.save_dir, scenario_id, "instances", "instances_info.json"
            )
            frame_instances_path = os.path.join(
                self.save_dir, scenario_id, "instances", "frame_instances.json"
            )

            with open(instances_info_path, "w") as f:
                json.dump(instances_info, f, indent=4)
            with open(frame_instances_path, "w") as f:
                json.dump(frame_instances, f, indent=4)
            
        

    def save_images(self, scenario_id, frame_idx):
        """Save images from all viewpoints and cameras.

        Camera ID assignment:
        0-5:   ego_vehicle (Front, FrontLeft, FrontRight, Back, BackLeft, BackRight)
        6-11:  ego_vehicle_behind
        12-17: other_vehicle
        18-23: other_vehicle_behind
        24-29: infrastructure
        """
        for vp_idx, viewpoint in enumerate(self.viewpoints):
            for cam_idx, camera_name in enumerate(self.camera_names):
                camera_id = vp_idx * 6 + cam_idx

                src_path = os.path.join(
                    self.load_dir, viewpoint, camera_name, scenario_id,
                    f"{scenario_id}_{frame_idx:03d}.jpg"
                )

                if os.path.exists(src_path):
                    dst_path = os.path.join(
                        self.save_dir, scenario_id, "images",
                        f"{frame_idx-1:03d}_{camera_id}.jpg"
                    )
                    # Copy image file
                    img = Image.open(src_path)
                    img.save(dst_path, 'JPEG')
            break  # Only save images from first viewpoint (ego_vehicle) 

    def save_calib(self, scenario_id, frame_idx):
        """Extract and save calibration data from PKL files.

        Intrinsics format: 3x3 matrix or [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        Extrinsics format: 4x4 transformation matrix
        """
        for vp_idx, viewpoint in enumerate(self.viewpoints):
            calib_file = os.path.join(
                self.load_dir, viewpoint, "calib", scenario_id,
                f"{scenario_id}_{frame_idx:03d}.pkl"
            )

            if not os.path.exists(calib_file):
                continue

            with open(calib_file, 'rb') as f:
                calib_data = pickle.load(f)
            self.calib_data = calib_data
            # Save lidar_to_ego transformation matrix for each frame
            lidar2ego_key = f"lidar_to_ego"
            if lidar2ego_key in calib_data:
                lidar_to_ego = calib_data[lidar2ego_key]
                lidar_to_ego_path = os.path.join(
                    self.save_dir, scenario_id, "lidar_to_ego",
                    f"{frame_idx-1:03d}.txt"
                )
                np.savetxt(lidar_to_ego_path, lidar_to_ego)
            # Save intrinsics and extrinsics for each camera
            for cam_idx, camera_name in enumerate(self.camera_names):
                camera_id = vp_idx * 6 + cam_idx

                # Get intrinsic matrix 
                intrinsic_key = f"intrinsic_{camera_name}"
                if intrinsic_key in calib_data:
                    intrinsic = calib_data[intrinsic_key]
                    # 列：012->120
                    # intrinsic = intrinsic[:, [1, 2, 0]]
                    intrinsic_path = os.path.join(
                        self.save_dir, scenario_id, "intrinsics",
                        f"{camera_id}.txt"
                    )
                    # Save as 3x3 matrix
                    np.savetxt(intrinsic_path, intrinsic)

                # Get extrinsic matrix (camera to ego transformation).
                lidar2ego_key = f"lidar_to_ego"
                lidar2cam_key = f"lidar_to_{camera_name}"
                if lidar2ego_key in calib_data and lidar2cam_key in calib_data:
                    extrinsic = calib_data[lidar2ego_key] @ np.linalg.inv(calib_data[lidar2cam_key])
                    extrinsic_path = os.path.join(
                        self.save_dir, scenario_id, "extrinsics",
                        f"{camera_id}.txt"
                    )
                    np.savetxt(extrinsic_path, extrinsic)
            break  # Only save calib from first viewpoint (ego_vehicle)

    def save_pose(self, scenario_id, frame_idx):
        """Extract ego vehicle pose from calibration data.

        We save ego_to_world transformation matrix for ego_vehicle frames.
        """
        # Only process ego_vehicle and ego_vehicle_behind for poses
        for vp_idx in [0, 1]:  # ego_vehicle, ego_vehicle_behind
            viewpoint = self.viewpoints[vp_idx]
            calib_file = os.path.join(
                self.load_dir, viewpoint, "calib", scenario_id,
                f"{scenario_id}_{frame_idx:03d}.pkl"
            )

            if not os.path.exists(calib_file):
                continue

            with open(calib_file, 'rb') as f:
                calib_data = pickle.load(f)

            if "ego_to_world" in calib_data:
                ego_pose = calib_data["ego_to_world"] # ego pose is ego_to_world
                pose_path = os.path.join(
                    self.save_dir, scenario_id, "ego_pose",
                    f"{frame_idx-1:03d}.txt"
                )
                np.savetxt(pose_path, ego_pose)
                break  # Only save one ego pose per frame

    def save_lidar(self, scenario_id, frame_idx):
        """Extract and save LiDAR point cloud data.

        LiDAR data is stored as NPZ format and converted to binary.
        """
        for viewpoint in self.viewpoints:
            lidar_file = os.path.join(
                self.load_dir, viewpoint, "lidar01", scenario_id,
                f"{scenario_id}_{frame_idx:03d}.npz"
            )

            if not os.path.exists(lidar_file):
                continue

            try:
                lidar_data = np.load(lidar_file)
                # NPZ contains 'data' key with point cloud
                if 'data' in lidar_data:
                    points = lidar_data['data']
                else:
                    # Try to get first available key
                    points = lidar_data[list(lidar_data.keys())[0]]

                lidar_path = os.path.join(
                    self.save_dir, scenario_id, "lidar",
                    f"{frame_idx-1:03d}.bin"
                )
                # Save as binary
                points.astype(np.float32).tofile(lidar_path)
                break  # Only save first available LiDAR(ego_vehicle lidar) 
            except Exception as e:
                print(f"Error loading lidar {lidar_file}: {e}")
                continue

    def save_dynamic_mask(self, scenario_id, frame_idx, class_valid='all'):
        """Generate and save dynamic object masks for ego_vehicle.

        Args:
            scenario_id (str): Scenario ID to process.
            frame_idx (int): Frame index to process.
            class_valid (str): Class filter ('all', 'human', 'vehicle').
        """
        assert class_valid in ['all', 'human', 'vehicle'], "Invalid class valid"

        # Define valid classes
        DA_DYNAMIC_CLASSES = ['car', 'van', 'truck', 'motorcycle', 'pedestrian', 'cyclist']
        DA_HUMAN_CLASSES = ['pedestrian', 'cyclist', 'motorcycle']
        DA_VEHICLE_CLASSES = ['car', 'van', 'truck']
        if class_valid == 'all':
            VALID_CLASSES = DA_DYNAMIC_CLASSES
        elif class_valid == 'human':
            VALID_CLASSES = DA_HUMAN_CLASSES
        elif class_valid == 'vehicle':
            VALID_CLASSES = DA_VEHICLE_CLASSES

        # Paths
        pose_path = f"{self.save_dir}/{scenario_id}/ego_pose/{frame_idx-1:03d}.txt"
        extrinsics_path = f"{self.save_dir}/{scenario_id}/extrinsics"
        intrinsics_path = f"{self.save_dir}/{scenario_id}/intrinsics"

        # Load ego pose
        ego_pose = np.loadtxt(pose_path)

        # Label path
        label_path = f"{self.load_dir}/ego_vehicle/label/{scenario_id}/{scenario_id}_{frame_idx:03d}.txt"

        # Read label file
        if not os.path.exists(label_path):
            print(f"Label file {label_path} not found, skipping...")
            return

        with open(label_path, 'r') as f:
            lines = f.readlines()

        # Parse ego velocity (first line)
        ego_velocity = np.array([float(v) for v in lines[0].strip().split()])

        # Parse object labels 
        objects = []# 对应场景视点时间帧
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 13:
                continue
            # 若要将carla左手系转换到诸如waymo的右手系，y轴、yaw、vy取反，转换矩阵做y轴翻转
            obj = {
                'type': parts[0],
                'center': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),  # Y可反转：carla左手系→右手系
                'size': np.array([float(parts[4]), float(parts[5]), float(parts[6])]),
                'yaw': float(parts[7]),  # 航向角可随Y反转而反转
                'velocity': np.array([float(parts[8]),float(parts[9])]),  # vy可随Y反转而反转
                'id': int(parts[10]),
                'lidar_points': int(parts[11]),
                'visible': parts[12].lower() == 'true'
            }
            objects.append(obj)

        # Process each camera 在每个视角处理当前时间帧的每个实例，生成当前时间帧当前视角动态mask
        for cam_idx in range(6):
            # Load camera parameters
            lidar2cam_key = f"lidar_to_{self.camera_names[cam_idx]}"
            lidar_to_cam = self.calib_data[lidar2cam_key]
            extrinsics = np.loadtxt(f"{extrinsics_path}/{cam_idx}.txt")
            intrinsics = np.loadtxt(f"{intrinsics_path}/{cam_idx}.txt")
            img_path = f"{self.save_dir}/{scenario_id}/images/{frame_idx-1:03d}_{cam_idx}.jpg"
            img = np.array(Image.open(img_path))
            img_h, img_w = img.shape[:2]
            # Initialize dynamic mask
            dynamic_mask = np.zeros_like(img, dtype=np.float32)[..., 0]
            # 调试：可视化3D框顶点投影点
            output_path = f"{self.save_dir}/{scenario_id}/images_projected_points_all/{frame_idx-1:03d}_{cam_idx}.jpg"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)# 目录不存在，创建目录；目录存在，继续向下执行
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)# 将RGB图像转换为BGR格式，以便在OpenCV中正确显示颜色(OpenCV 使用的是 BGR 颜色通道顺序，而 PIL 和大多数其他库使用的是 RGB 颜色通道顺序)
            for obj in objects:# 处理当前时间帧每个实例
                if obj['type'] not in VALID_CLASSES: 
                    continue
                # 判断实例在该视点时间帧的有效性
                close_objects_thre = 10 # 车辆视点：10m阈值
                if np.sqrt(obj['center'][0] ** 2 + obj['center'][1] ** 2 + obj['center'][2] ** 2) < close_objects_thre:# 若实例中心到雷达原点的直线距离小于阈值，则认为是近距离实例，取1
                    close_objects_flag=1# 近距离实例标识
                else:# 否则认为是远距离实例，取0
                    close_objects_flag=0
                if obj['type'] == 'pedestrian' or obj['type'] == 'motorcycle' \
                        or obj['type'] == 'cyclist':# 实例类型是行人/骑摩托车的人/骑自行车的人
                    if close_objects_flag:# 近距离实例
                        if obj['lidar_points'] < 2 and not obj['visible']:# 激光雷达点数大于等于2(优先，若点数足够，就有效)或者相机可见该实例，则认为该实例有效，否则认为无效
                            continue
                    else:# 远距离实例
                        if not obj['visible'] or obj['lidar_points'] < 1:# 相机可见该实例且激光雷达点数大于等于1，则认为该实例有效，否则认为无效
                            continue
                else:# 实例类型是汽车、卡车、货车
                    if close_objects_flag:# 近距离实例
                        if obj['lidar_points'] < 5 and not obj['visible']:# 激光雷达点数大于等于5(优先，若点数足够，就有效)或者相机可见该实例，则认为该实例有效，否则认为无效
                            continue
                    else:# 远距离实例
                        if not obj['visible'] or obj['lidar_points'] < 2:# 相机可见该实例且激光雷达点数大于等于2，则认为该实例有效，否则认为无效
                            continue
                # Check velocity
                speed = np.linalg.norm(obj['velocity'])# L2norm
                # if speed <= 0:# Current frame velocity is zero, considered a static object — skipping this instance
                #     continue

                # Transform object center and corners to camera coordinates
                center = obj['center']
                size = obj['size']
                yaw = obj['yaw']

                # Compute 3D bounding box corners in object coordinates逆时针顺序 实例框中心在几何中心
                corners = np.array([
                    [-size[0] / 2, -size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, size[1] / 2, -size[2] / 2],
                    [size[0] / 2, size[1] / 2, -size[2] / 2],
                    [size[0] / 2, -size[1] / 2, -size[2] / 2],
                    [-size[0] / 2, -size[1] / 2, size[2] / 2],
                    [-size[0] / 2, size[1] / 2, size[2] / 2],
                    [size[0] / 2, size[1] / 2, size[2] / 2],
                    [size[0] / 2, -size[1] / 2, size[2] / 2]
                ])
                # 实例框中心移到底部中心
                # corners = np.array([
                #     [-size[0] / 2, -size[1] / 2, 0],
                #     [-size[0] / 2, size[1] / 2, 0],
                #     [size[0] / 2, size[1] / 2, 0],
                #     [size[0] / 2, -size[1] / 2, 0],
                #     [-size[0] / 2, -size[1] / 2, size[2]],
                #     [-size[0] / 2, size[1] / 2, size[2]],
                #     [size[0] / 2, size[1] / 2, size[2]],
                #     [size[0] / 2, -size[1] / 2, size[2]]
                # ])
                # center += np.array([0, 0, -size[2]/2])
                # Rotate and translate corners from object coordinates to lidar coordinates
                rotation = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]
                ])
                corners = corners @ rotation.T + center

                # Transform from lidar coordinates to camera coordinates
                corners = (np.column_stack((corners, np.ones(8))) @ lidar_to_cam.T)[:, :3]# (x, y, z)-x向前、y向右、z向上，x相当于标准情况下z

                # Check depth 保证所有角点在相机前方（x>0），否则跳过该实例，以此保证实例在相机视角前方
                if not np.all(corners[:, 0] > 0):# 相机坐标系x和像素齐次坐标x相同，都是深度
                    continue
                
                # Transform from camera coordinates to pixel coordinates
                pixels = corners @ intrinsics.T
                pixels = pixels[:, :2] / pixels[:, 2:3]# 齐->非齐 

                # 计算通过宽高裁剪的比例
                x_min, y_min = np.min(pixels, axis=0)# 最小的x和y
                x_max, y_max = np.max(pixels, axis=0)# 最大的x和y
                original_area = (x_max - x_min) * (y_max - y_min)# 2D框初始面积
                x_min, x_max = np.clip(x_min, 0, img_w), np.clip(x_max, 0, img_w)
                y_min, y_max = np.clip(y_min, 0, img_h), np.clip(y_max, 0, img_h)
                # Clip box to image bounds.横坐标在[0, img_w]范围内，纵坐标在[0, img_h]范围内
                pixels[:, 0] = np.clip(pixels[:, 0], 0, img.shape[1])
                pixels[:, 1] = np.clip(pixels[:, 1], 0, img.shape[0])
                new_area = (x_max - x_min) * (y_max - y_min)# 2D框在图像内部的面积
                clip_large = new_area / original_area < 1/3# 2D框在图像内部的面积小于初始面积的1/3，认为该实例被裁剪过于严重，排除该实例
                
                
                # 排除投影到图像平面上是线段（长或宽为0）、裁剪区域过大的实例
                if x_max - x_min == 0 or y_max - y_min == 0 or clip_large:
                    continue
                
                # 调试：可视化3D框顶点投影点
                if class_valid == 'all':
                    for pixel in pixels:
                        cv2.circle(img_bgr, (int(pixel[0]), int(pixel[1])), radius=3, color=(0, 255, 0), thickness=-1)
                    cv2.imwrite(output_path, img_bgr)
                

                # Draw projected 2D box onto the image. 
                xy = (pixels[:, 0].min(), pixels[:, 1].min())
                width = pixels[:, 0].max() - pixels[:, 0].min()
                height = pixels[:, 1].max() - pixels[:, 1].min()
                # max pooling 在实例的2D边界框区域内，动态mask的值为该实例的速度（m/s）；如果多个实例的边界框区域有重叠，动态mask取重叠区域内所有实例速度的最大值
                dynamic_mask[
                    int(xy[1]) : int(xy[1] + height),
                    int(xy[0]) : int(xy[0] + width),
                ] = np.maximum(
                    dynamic_mask[
                        int(xy[1]) : int(xy[1] + height),
                        int(xy[0]) : int(xy[0] + width),
                    ],
                    speed,
                )
            # thresholding, use 1.0 m/s to determine whether the pixel is moving
            dynamic_mask = np.clip((dynamic_mask > 0.0) * 255, 0, 255).astype(np.uint8)# 动态mask二值化，速度大于0.0 m/s的像素被认为是动态的，动态像素值为255，静态像素值为0
            dynamic_mask = Image.fromarray(dynamic_mask, "L")# 将动态mask保存为灰度图像，像素值为0或255
            # Save mask
            dynamic_mask_dir = os.path.join(self.save_dir, scenario_id, "dynamic_masks", class_valid)
            os.makedirs(dynamic_mask_dir, exist_ok=True)
            dynamic_mask_path = os.path.join(dynamic_mask_dir, f"{frame_idx-1:03d}_{cam_idx}.png")# 
            dynamic_mask.save(dynamic_mask_path)
        return objects# 对应场景视点时间帧

    def save_objects(self, scenario_id):
        """Extract and save object annotations.

        Parses label files to extract 3D bounding boxes and tracking information.

        Label file format (per line):
        object_type x y z length width height yaw vx vy object_id lidar_numbers visibility
        """
        instances_info = {}
        frame_instances = {}

        # Parse label files from first viewpoint (ego_vehicle)
        viewpoint = self.viewpoints[0]
        label_dir = os.path.join(self.load_dir, viewpoint, "label", scenario_id)

        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

        for label_file in label_files:# 遍历该场景所有时间帧
            frame_idx = int(label_file.split('_')[-1].replace('.txt', '')) - 1
            frame_instances[str(frame_idx)] = []

            label_path = os.path.join(label_dir, label_file)
            pose_path = f"{self.save_dir}/{scenario_id}/ego_pose/{frame_idx:03d}.txt"
            
            # Load ego pose
            ego_pose = np.loadtxt(pose_path)
        
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # Parse object lines
            for line in lines[1:]:# 遍历该时间帧下所有实例
                parts = line.strip().split()
                if len(parts) < 13:
                    continue
                # Parse: type x y z length width height yaw vx vy object_id lidar_numbers visibility
                obj_type = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                length, width, height = float(parts[4]), float(parts[5]), float(parts[6])
                yaw = float(parts[7])
                vx, vy = float(parts[8]), float(parts[9])
                object_id = int(parts[10])
                lidar_numbers = int(parts[11])
                
                str_id = str(object_id)
                # 在frame_instances中添加实例id
                frame_instances[str(frame_idx)].append(str_id)
                
                if str_id not in instances_info:
                    instances_info[str_id] = dict(
                        id=str_id,
                        # class_ind=l.type,
                        class_name=obj_type,
                        frame_annotations={
                            "frame_idx": [],
                            "obj_to_world": [],
                            "box_size": [],
                        }
                    )
                
                # 构造当前时间帧该实例在世界坐标系的位姿 
                c = np.math.cos(yaw)
                s = np.math.sin(yaw)
                o2lidar = np.array([
                    [ c, -s,  0, x],
                    [ s,  c,  0, y],
                    [ 0,  0,  1, z],
                    [ 0,  0,  0,  1]])# 当前时间帧该实例在雷达坐标系的位姿 
                o2w = ego_pose @ self.calib_data["lidar_to_ego"] @ o2lidar
                
                instances_info[str_id]["frame_annotations"]["frame_idx"].append(frame_idx)
                instances_info[str_id]["frame_annotations"]["box_size"].append([length, width, height])
                instances_info[str_id]["frame_annotations"]["obj_to_world"].append(o2w.tolist())
                
        # Correct ID mapping 将实例id与序号对应，从0开始
        id_map = {}
        for i, (k, v) in enumerate(instances_info.items()):
            id_map[v["id"]] = i

        # Update keys in instances_info 更新实例信息字典：将实例id键变成序号
        new_instances_info = {}
        for k, v in instances_info.items():
            new_instances_info[id_map[v["id"]]] = v

        # Update keys in frame_instances 更新时间帧实例字典：将实例id值变成序号
        new_frame_instances = {}
        for k, v in frame_instances.items():
            new_frame_instances[k] = [id_map[i] for i in v]
        
        return new_instances_info, new_frame_instances
