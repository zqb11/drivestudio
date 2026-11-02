"""Spatial identity propagation adapted from DIAL-GS stage two.

Unlike the unified PVG field, DriveStudio has explicit background and instance
nodes. Only reliable instance points teach nearby background points whose
projected pixel label agrees with that instance. Neighbor selection is detached;
the KL divergence retains gradients to both identity predictions.
"""

import torch
import torch.nn.functional as F

# 近邻搜索用于选择配对
@torch.no_grad()
def _nearest_neighbors(means, anchors, candidates, k, max_distance,
                       anchor_chunk_size, candidate_chunk_size):
    """Yield global index pairs without allocating the full distance matrix."""
    k = min(k, candidates.numel())# 候选不足 k 个时，缩小 k
    for anchor_start in range(0, anchors.numel(), anchor_chunk_size):# 分批处理锚点
        anchor_ids = anchors[anchor_start:anchor_start + anchor_chunk_size]# 当前批次锚点的索引
        distances = means.new_full((anchor_ids.numel(), k), float("inf"))# 为每个锚点创建 k 个“当前最近距离”，初始为无穷大
        neighbors = torch.full_like(distances, -1, dtype=torch.long)# 创建对应邻居候选索引，初始 -1 表示尚未找到
        for candidate_start in range(0, candidates.numel(), candidate_chunk_size):# 分批处理候选
            candidate_ids = candidates[candidate_start:candidate_start + candidate_chunk_size]# 当前批次候选的索引
            current_distances = torch.cdist(
                means[anchor_ids].float(), means[candidate_ids].float(),
                compute_mode="donot_use_mm_for_euclid_dist",
            )# 计算当前批次锚点与候选中心的欧氏距离
            current_distances.masked_fill_(anchor_ids[:, None] == candidate_ids[None, :], float("inf"))# 近邻通用保护，不会出现：如果两端是同一个高斯，将距离设为无穷大，排除自身
            current_ids = candidate_ids[None, :].expand(anchor_ids.numel(), -1)# 为每个锚点展开一样的候选索引
            merged_distances = torch.cat((distances, current_distances), dim=1)# 合并此前最近距离和当前批次距离
            merged_neighbors = torch.cat((neighbors, current_ids), dim=1)# 以完全相同的顺序合并候选索引
            distances, indices = merged_distances.topk(k, dim=1, largest=False)# 每个锚点只保留最小的 k 个距离
            neighbors = merged_neighbors.gather(1, indices)# 最小k个距离对应的候选索引
        valid = (neighbors >= 0) & (distances <= max_distance)# 邻居候选距离不超过0.5，才认为有效
        if valid.any():# 为true，存在锚点与候选的配对时，输出锚点索引、与其配对的候选索引
            yield anchor_ids[:, None].expand_as(neighbors)[valid], neighbors[valid]

# 3d id loss
def spatial_id_kl(means, logits, instance_ids, opacities,
                  pixel_labels_at_points, cfg=None):
    """Propagate reliable dynamic IDs to nearby background Gaussians.

    Args:
        means: Current world positions, ``(N, 3)``.
        logits: Shared classifier logits, ``(N, C)``. Class 0 is background.
        instance_ids: Structural node identity, ``(N,)``; 0 is background.
        opacities: Activated per-point opacity, ``(N,)`` or ``(N, 1)``.
        pixel_labels_at_points: Projected labels, ``(N,)``; -1 is invalid,
            0 is static, and positive labels are global instance classes.
        cfg: Mapping with optional confidence (0.8), opacity_threshold (0.01),
            max_distance (0.5 metres), k (10), max_anchors (2048),
            max_candidates (20000), anchor_chunk_size (128), and
            candidate_chunk_size (2048).

    Returns:
        ``(loss, stats)``: mean KL(anchor || candidate) divided by C, and
        detached integer counts for logging separately from training losses.
        Empty selections return a zero connected to logits for backward().
    """
    cfg = {} if cfg is None else cfg
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if logits.ndim != 2 or logits.shape[0] != means.shape[0] or logits.shape[1] < 1:
        raise ValueError("logits must have shape (N, C) with C >= 1")
    num_points = means.shape[0]
    if any(value.numel() != num_points for value in (instance_ids, opacities, pixel_labels_at_points)):
        raise ValueError("point identity, opacity and pixel labels must contain N values")
    k = int(cfg.get("k", 10)) # 每个节点高斯锚点最多选择 10 个背景高斯候选
    max_anchors = int(cfg.get("max_anchors", 2048)) # 全场景最多保留 2048 个节点高斯锚点
    max_candidates = int(cfg.get("max_candidates", 20000))# 全场景最多保留 20000 个背景高斯候选
    anchor_chunk_size = int(cfg.get("anchor_chunk_size", 128))
    candidate_chunk_size = int(cfg.get("candidate_chunk_size", 2048))
    max_distance = float(cfg.get("max_distance", 0.5))# 邻居距离、搜索半径不得超过 0.5 米
    if min(k, max_anchors, max_candidates, anchor_chunk_size, candidate_chunk_size) < 1:
        raise ValueError("neighbor count, sample limits and chunk sizes must be positive")
    if max_distance <= 0:
        raise ValueError("max_distance must be positive")

    instance_ids = instance_ids.reshape(-1)
    pixel_labels_at_points = pixel_labels_at_points.reshape(-1)
    with torch.no_grad():
        probabilities = logits.detach().float().softmax(dim=-1)# 高斯的归属id概率
        confidence, predictions = probabilities.max(dim=-1)# 高斯的归属id概率最大值及其在概率张量中的索引（高斯的预测归属id）
        exists = opacities.detach().reshape(-1) > float(cfg.get("opacity_threshold", 0.01))# 保留不透明度大于 0.01 的高斯
        finite_means = means.detach().isfinite().all(dim=-1)
        anchors = torch.where(
            exists & finite_means & (instance_ids > 0) # 透明度有效、坐标有效，并且目标归属于动态节点
            & (predictions == instance_ids)# 分类器预测的高斯归属id与目标归属id一致，确保高斯的归属id概率正确，高斯可以正确学习id
            & (confidence >= float(cfg.get("confidence", 0.8)))# # 预测置信度至少为 0.8
            & (pixel_labels_at_points == instance_ids)# 中心投影的像素弱标签与目标归属id一致，确保当前投影的像素弱标签也支持同一id
        )[0]# 节点高斯锚点索引
        candidates = torch.where(
            exists & finite_means & (instance_ids == 0)# 透明度有效、坐标有效，并且目标归属于静态背景=明显显现的高斯-节点高斯
            & (pixel_labels_at_points > 0)# 像素投影弱标签属于节点
            & (pixel_labels_at_points < logits.shape[1])
        )[0]# 背景高斯候选索引（如：背景中的节点残影高斯、车辆后方背景高斯）
        if anchors.numel() > max_anchors:# 节点高斯锚点过多时，随机打乱索引，只保留上限数量
            anchors = anchors[torch.randperm(anchors.numel(), device=anchors.device)[:max_anchors]]
        if candidates.numel() > max_candidates:# 背景高斯候选过多时，同样随机采样，只保留上限数量
            candidates = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:max_candidates]]

    stats = {
        "num_anchors": anchors.numel(),# 节点高斯锚点数量
        "num_candidates": candidates.numel(),# 背景高斯候选数量
        "num_pairs": 0,# 节点高斯锚点—背景高斯候选配对数，初始化为零
    }
    loss = logits.sum() * 0.0# 初始化一个值为零、但仍连接分类分数计算图的损失，即使后面没有有效配对，也可以安全调用 backward()
    if anchors.numel() == 0 or candidates.numel() == 0:
        return loss, stats

    log_probabilities = F.log_softmax(logits.float(), dim=-1)# 取对数的高斯id概率
    for instance_id in instance_ids[anchors].unique():# 遍历节点高斯锚点归属的节点id
        instance_anchors = anchors[instance_ids[anchors] == instance_id]# 属于当前节点的锚点
        instance_candidates = candidates[pixel_labels_at_points[candidates] == instance_id]# 投影到当前节点区域的背景高斯候选
        if instance_candidates.numel() == 0:# 当前节点没有匹配候选时，处理下一个节点
            continue
        for anchor_ids, candidate_ids in _nearest_neighbors(
            means.detach(), instance_anchors, instance_candidates, k, max_distance,
            anchor_chunk_size, candidate_chunk_size,
        ):# 对每批锚点寻找邻居候选后，输出锚点索引、对应候选索引
            anchor_log_probs = log_probabilities[anchor_ids]# 配对锚点的id对数概率
            candidate_log_probs = log_probabilities[candidate_ids]# 配对候选的id对数概率
            loss = loss + (anchor_log_probs.exp() * (anchor_log_probs - candidate_log_probs)).sum()# 累加每个批次的锚点和候选id概率分布的KL散度
            stats["num_pairs"] += anchor_ids.numel()# 累加每个批次的配对数量
    if stats["num_pairs"]:
        loss = loss / (stats["num_pairs"])# 所有配对的锚点和候选id概率分布的KL散度/配对数量，得到平均每个配对的KL
    return loss, stats
