from typing import List, Tuple
import torch

from .pixel_source import ScenePixelSource

class SplitWrapper(torch.utils.data.Dataset):

    # a sufficiently large number to make sure we don't run out of data
    _num_iters = 1000000

    def __init__(
        self,
        datasource: ScenePixelSource,
        split_indices: List[int] = None,
        split: str = "train",
    ):
        super().__init__()
        self.datasource = datasource
        self.split_indices = split_indices
        self.split = split
    # 获得图像帧索引对应的下采样目标图像和相机信息，之后重置下采样因子
    def get_image(self, idx, camera_downscale) -> dict:
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
        self.datasource.update_downscale_factor(downscale_factor)# 更新下采样因子
        image_infos, cam_infos = self.datasource.get_image(self.split_indices[idx])# 图像帧索引对应的目标图像和相机信息
        self.datasource.reset_downscale_factor()# 重置下采样因子
        return image_infos, cam_infos
    # 根据随机选取的训练图像帧索引得到对应的下采样目标图像和相机信息，之后再重置下采样因子
    def next(self, camera_downscale) -> Tuple[dict, dict]:
        assert self.split == "train", "Only train split supports next()"
        
        img_idx = self.datasource.propose_training_image(
            candidate_indices=self.split_indices
        )# 从训练图像帧索引中随机选取训练图像帧
        
        downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor# 下采样因子
        self.datasource.update_downscale_factor(downscale_factor)# 更新下采样因子
        image_infos, cam_infos = self.datasource.get_image(img_idx)# 根据图像帧索引得到对应的目标图像和相机信息
        self.datasource.reset_downscale_factor()# 重置下采样因子
        
        return image_infos, cam_infos
    
    def __getitem__(self, idx) -> dict:
        return self.get_image(idx, camera_downscale=1.0)

    def __len__(self) -> int:
        return len(self.split_indices)

    @property
    def num_iters(self) -> int:
        return self._num_iters

    def set_num_iters(self, num_iters) -> None:
        self._num_iters = num_iters
