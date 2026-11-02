"""Conservative instance labels and loss for rendered identity features."""

import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

# 像素弱id
@torch.no_grad()
def project_instance_labels(
    dynamic_mask: Tensor,
    sky_mask: Optional[Tensor],
    egocar_mask: Optional[Tensor],
    camtoworld: Tensor,
    intrinsics: Tensor,
    instance_poses: Tensor,
    instance_sizes: Tensor,
    instance_valid: Tensor,
    near_plane: float = 0.1,
    chunk_size: int = 65536,
) -> Tensor:
    """Assign labels using pixel-center rays intersecting oriented 3D boxes.

    Inputs use OpenCV cameras (positive z forward). ``intrinsics`` must already
    match the H x W masks, and poses are object-to-world matrices in the same
    world frame as ``camtoworld``. Sizes are full object lengths, not extents.
    Label 0 denotes static pixels, i + 1 denotes box i, and -1 is ignored.
    Only dynamic pixels hitting exactly one valid box wholly beyond the near
    plane receive an instance label. Overlap, missing boxes, and boxes crossing
    the near plane are ignored. Sky and ego masks always take precedence.

    This is box-derived supervision, not a visible-surface instance annotation:
    it intentionally ignores overlapping boxes instead of guessing occlusion.
    Work is chunked over dynamic pixels without allocating an H x W x M tensor.
    """
    if dynamic_mask.ndim != 2:
        raise ValueError("dynamic_mask must have shape [H, W]")
    if chunk_size <= 0 or not math.isfinite(near_plane) or near_plane <= 0:
        raise ValueError("chunk_size and near_plane must be positive")
    num_instances = instance_poses.shape[0]
    if instance_poses.shape != (num_instances, 4, 4):
        raise ValueError("instance_poses must have shape [M, 4, 4]")
    if instance_sizes.shape != (num_instances, 3):
        raise ValueError("instance_sizes must have shape [M, 3]")
    if instance_valid.shape != (num_instances,):
        raise ValueError("instance_valid must have shape [M]")
    if camtoworld.shape != (4, 4) or intrinsics.shape != (3, 3):
        raise ValueError("camera matrices must have shapes [4, 4] and [3, 3]")

    device = dynamic_mask.device
    dtype = torch.float64 if camtoworld.dtype == torch.float64 else torch.float32
    camtoworld = camtoworld.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)
    poses = instance_poses.to(device=device, dtype=dtype)
    sizes = instance_sizes.to(device=device, dtype=dtype)
    valid = instance_valid.to(device=device).bool()
    valid &= torch.isfinite(poses).all(dim=-1).all(dim=-1)
    valid &= torch.isfinite(sizes).all(dim=-1) & (sizes > 0).all(dim=-1)

    dynamic = dynamic_mask > 0.5# 动态掩码布尔值
    ignore = torch.zeros_like(dynamic)# 初始化为全false的忽略掩码
    for mask in (sky_mask, egocar_mask):
        if mask is not None:
            if mask.shape != dynamic_mask.shape:# 天空和自车掩码必须与动态掩码形状匹配
                raise ValueError("sky_mask and egocar_mask must match dynamic_mask")
            ignore |= mask.to(device=device) > 0.5# 忽略掩码中天空和自车区域为true
    labels = torch.zeros_like(dynamic_mask, dtype=torch.long)# 像素的目标id图，初始化为全0，为背景像素赋予id=0
    labels[dynamic | ignore] = -1 # 目标id图中动态、天空、自车像素的id暂时设为-1
    pixels = torch.nonzero(dynamic & ~ignore, as_tuple=False)# 获取既属于动态区域，又不属于天空或自车的像素坐标，对这些像素构造射线，判断它们命中哪个节点框
    box_indices = torch.nonzero(valid, as_tuple=False).flatten()# 取得当前时间步有效节点索引
    if pixels.numel() == 0 or box_indices.numel() == 0:
        return labels

    camera_rotation = camtoworld[:3, :3]
    origin = camtoworld[:3, 3]# 世界空间光心
    inverse_intrinsics = torch.linalg.inv(intrinsics)
    # Camera-space support radius along z gives the nearest box corner depth.
    relative_centers = (poses[:, :3, 3] - origin) @ camera_rotation
    relative_rotations = camera_rotation.T.unsqueeze(0) @ poses[:, :3, :3]
    nearest_depth = relative_centers[:, 2] - (
        relative_rotations[:, 2].abs() * sizes * 0.5
    ).sum(dim=-1)
    beyond_near = nearest_depth > near_plane
    # 从动态像素发出射线，判断它命中哪个节点框，则赋予它命中节点的id
    for start in range(0, pixels.shape[0], chunk_size):# 分批处理目标图像的动态像素
        coords = pixels[start:start + chunk_size]# 当前批次的动态像素坐标
        pixel_centers = torch.stack((
            coords[:, 1].to(dtype) + 0.5,
            coords[:, 0].to(dtype) + 0.5,
            torch.ones(coords.shape[0], device=device, dtype=dtype),
        ), dim=-1)# 动态像素中心
        camera_directions = pixel_centers @ inverse_intrinsics.T# 相机空间像素射线方向
        # Normalize to camera z = 1, so ray distances are camera depths.
        camera_directions = camera_directions / camera_directions[:, 2:3]# z归一化的相机空间像素射线方向
        directions = camera_directions @ camera_rotation.T# 世界空间像素射线方向
        hit_count = torch.zeros(coords.shape[0], dtype=torch.long, device=device)# 像素射线命中节点框的数量
        instance_ids = torch.full_like(hit_count, -1)# 像素射线命中节点的id
        for box_index in box_indices:# 遍历当前时间步有效的节点框
            rotation = poses[box_index, :3, :3]
            local_origin = (origin - poses[box_index, :3, 3]) @ rotation# 局部空间光心
            local_directions = directions @ rotation# 局部空间像素射线方向
            half_size = sizes[box_index] * 0.5
            parallel = local_directions.abs() < 1e-8# 识别哪个方向分量为0
            safe_directions = torch.where(
                parallel, torch.ones_like(local_directions), local_directions
            )# 把值为0的方向分量替换为1，避免除以0
            t1 = (-half_size - local_origin) / safe_directions# 射线与节点框xyz轴 负 平面相交时的深度
            t2 = (half_size - local_origin) / safe_directions# 射线与节点框xyz轴 正 平面相交时的深度
            enter = torch.minimum(t1, t2).masked_fill(parallel, -torch.inf)# 射线与节点框xyz轴 正负 平面相交时的进入深度
            leave = torch.maximum(t1, t2).masked_fill(parallel, torch.inf)# 射线与节点框xyz轴 正负 平面相交时的离开深度
            parallel_outside = (parallel & (local_origin.abs() > half_size)).any(dim=-1)# 排除平行于三轴任意一轴且光心在框外的射线
            near = enter.max(dim=-1).values.clamp_min(near_plane)# 射线进入节点框的深度
            far = leave.min(dim=-1).values# 射线离开节点框的深度
            hits = (far >= near) & ~parallel_outside# 像素射线是否命中当前节点框
            # Cross-near boxes still count as ambiguity blockers, so another
            # overlapping box cannot incorrectly label their pixels.
            hit_count += hits.long()# 像素射线命中节点框的数量
            instance_ids[hits & beyond_near[box_index]] = box_index + 1# 给 当前节点框整体在近剪裁面之后 且 命中节点框的 像素 赋予当前节点框的id，id从1开始
        unambiguous = (hit_count == 1) & (instance_ids > 0)# 接受只命中一个节点框且该节点框在近剪裁面之后的像素，这些有效像素位置为true
        labels[coords[unambiguous, 0], coords[unambiguous, 1]] = instance_ids[unambiguous]# 将有效像素命中的节点id写入目标id图
    return labels# 背景像素id=0，有效动态像素id=像素射线命中节点的id，没命中、命中多个框（无深度或可靠的节点id掩码）或唯一命中的框不满足裁剪条件的动态像素id=-1，自车、天空像素id=-1

# 2d id loss
def masked_id_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    opacity: Tensor,
    min_opacity: float = 0.05,
) -> Tensor:
    """Mean identity CE / log(C), ignoring invalid labels and low coverage.

    ``logits`` has shape [H, W, C], labels [H, W], and opacity [H, W] or
    [H, W, 1]. Only the opacity-based selection is detached; logits keep their
    complete gradient path through features, rasterization, and geometry.
    An empty selection returns a differentiable zero.
    """
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [H, W, C] and labels [H, W]")
    if logits.shape[-1] < 2:
        raise ValueError("identity classification requires at least two classes")
    if not math.isfinite(min_opacity) or not 0 <= min_opacity <= 1:
        raise ValueError("min_opacity must be in [0, 1]")
    if opacity.shape == labels.shape + (1,):# 将累计不透明度图的最后一维去掉，得到与目标id图相同形状的张量
        opacity = opacity[..., 0]
    if opacity.shape != labels.shape:
        raise ValueError("opacity must have shape [H, W] or [H, W, 1]")
    labels = labels.to(device=logits.device, dtype=torch.long)
    if ((labels < -1) | (labels >= logits.shape[-1])).any():
        raise ValueError("labels must be -1 or a valid class index")
    coverage = opacity.detach().to(device=logits.device)# 取得用于筛选的不透明度，不参与优化
    valid = (labels >= 0) & torch.isfinite(coverage) & (coverage >= min_opacity)# 2d id loss监督像素：背景和有效动态像素，累计不透明度不是NAN或无穷且大于等于最小不透明度阈值
    selected_logits = logits[valid]# 监督像素的id分数数组（未经softmax）（M, C）
    if selected_logits.shape[0] == 0:# 若没有监督像素，损失返回0
        return selected_logits.sum()
    return F.cross_entropy(selected_logits, labels[valid]) / math.log(logits.shape[-1])# 监督像素的目标id和id分数的交叉熵损失：先对id分数进行softmax得到id概率，再在id分数中取目标id的概率计算负对数，然后所有像素的损失取平均，最后除以log(C)归一化
