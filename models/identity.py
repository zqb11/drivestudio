"""Scene-wide identity metadata and classifier; Gaussian ownership stays unchanged."""
import torch
from torch import nn
from importlib import import_module

project_instance_labels = import_module("models.2d_id_loss").project_instance_labels
masked_id_cross_entropy = import_module("models.2d_id_loss").masked_id_cross_entropy
spatial_id_kl = import_module("models.3d_id_loss").spatial_id_kl


def id_3d_enabled(step, cfg):
    return cfg.get("w", 0.0) > 0 and step >= cfg.get("start_step", 25000)

# 全局id
class IdentityField(nn.Module):
    # 为背景和每个节点生成全局id
    def __init__(self, embedding_dim, node_dataset_ids, poses, sizes, valid, true_ids):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.node_dataset_ids = {k: list(v) for k, v in node_dataset_ids.items()}
        dataset_ids = [i for ids in self.node_dataset_ids.values() for i in ids]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("An instance cannot belong to two identity node classes")
        self.num_classes = len(dataset_ids) + 1# 全局id个数=len(节点连续id)+1（静态背景id）
        if self.num_classes < 2:
            raise ValueError("Identity supervision requires at least one dynamic instance")
        self.register_buffer("dataset_ids", torch.tensor(dataset_ids, dtype=torch.long, device=poses.device))
        self.register_buffer("poses", poses.detach().clone())
        self.register_buffer("sizes", sizes.detach().clone())
        self.register_buffer("valid", valid.detach().clone().bool())
        self.register_buffer("true_ids", true_ids.detach().clone())
        offset = 1
        for name, ids in self.node_dataset_ids.items():# 遍历节点类别，按节点类别为每个节点生成全局id，从1开始，0为静态背景id
            self.register_buffer("labels_" + name, torch.arange(offset, offset + len(ids), device=poses.device))
            offset += len(ids)
        self.classifier = nn.Linear(embedding_dim, self.num_classes).to(poses.device)# 创建分类器：MLP，输入为id特征，输出为取到各个全局id的分数（未经softmax）

    def get_extra_state(self):
        return {"embedding_dim": self.embedding_dim, "node_dataset_ids": self.node_dataset_ids}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Checkpoint identity mapping does not match initialized instances")

    @classmethod
    def from_checkpoint(cls, state, prefix="identity_field."):
        metadata = state[prefix + "_extra_state"]
        return cls(metadata["embedding_dim"], metadata["node_dataset_ids"],
                   state[prefix + "poses"], state[prefix + "sizes"],
                   state[prefix + "valid"], state[prefix + "true_ids"])
    # 生成背景和节点高斯的归属id张量
    def point_labels(self, class_name, model):
        if class_name == "Background":# 背景高斯的归属id零张量
            return torch.zeros(model.num_points, dtype=torch.long, device=model._means.device)
        return getattr(self, "labels_" + class_name)[model.point_ids.reshape(-1)]# 用节点高斯的节点内部id查询节点的全局id，得到节点高斯的归属id张量
    # 计算目标图像的目标id图
    @torch.no_grad()
    def image_labels(self, image_infos, cam_infos, frame):
        return project_instance_labels(
            dynamic_mask=image_infos["dynamic_masks"],
            sky_mask=image_infos.get("sky_masks"),
            egocar_mask=image_infos.get("egocar_masks"),
            camtoworld=cam_infos["camera_to_world"],
            intrinsics=cam_infos["intrinsics"],
            instance_poses=self.poses[frame], instance_sizes=self.sizes,
            instance_valid=self.valid[frame],
        )

    @staticmethod
    @torch.no_grad()
    def labels_at_points(means, labels, camera):
        w2c = torch.linalg.inv(camera.camtoworlds.detach())
        points = means.detach() @ w2c[:3, :3].T + w2c[:3, 3]
        projected = points @ camera.Ks.detach().T
        uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)# 高斯中心的像素坐标
        h, w = labels.shape
        inside = ((points[:, 2] > 0.1) & torch.isfinite(uv).all(-1)
                  & (uv[:, 0] >= 0) & (uv[:, 0] < w)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < h))# 保留有效高斯：位于相机前方，深度大于 0.1；像素坐标没有 NaN 或无穷值；投影位于图像范围内
        result = torch.full((means.shape[0],), -1, dtype=torch.long, device=means.device)# 为所有高斯的像素投影弱标签填入-1
        pixel = uv[inside].floor().long()# 有效高斯的像素坐标向下取整
        result[inside] = labels[pixel[:, 1], pixel[:, 0]]# 有效高斯的像素投影弱标签，无效高斯保持-1
        return result
    #id损失计算
    def compute_losses(self, outputs, image_infos, cam_infos, frame, step, cfg):
        labels = self.image_labels(image_infos, cam_infos, frame)# 计算目标图像的目标id图：弱标签，非目标id标注
        outputs["id_labels"] = labels
        losses = {}
        stats = {"valid_pixels": (labels >= 0).sum().item(),# 目标图像的背景和有效动态像素数
                 "dynamic_pixels": (labels > 0).sum().item()}# 目标图像的有效动态像素数
        cfg2d = cfg.get("id_2d", {})# 2D ID损失配置
        if cfg2d.get("w", 0.0) > 0:
            logits2d = self.classifier(outputs["id_features"])# 目标图像的id分数图（未经softmax）
            losses["id_2d_loss"] = cfg2d.w * masked_id_cross_entropy(
                logits2d, labels,
                outputs["opacity"], cfg2d.get("min_opacity", 0.05))# 2d id loss：监督像素的目标id和id分数的交叉熵损失
            with torch.no_grad():
                dynamic = labels > 0
                stats["dynamic_id_accuracy"] = (
                    (logits2d.argmax(-1)[dynamic] == labels[dynamic]).float().mean().item()
                    if dynamic.any() else 0.0)
        cfg3d = cfg.get("id_3d", {})# 3D ID损失配置
        if id_3d_enabled(step, cfg3d):# 从25K步开始启用3d id loss
            gs, camera = outputs["id_gaussians"], outputs["id_camera"]# 目标图像时间帧的高斯属性和目标图像相机信息
            loss3d, stats3d = spatial_id_kl(# 输入参数：世界空间高斯中心、高斯的归属id分数、高斯的目标归属id、高斯不透明度、有效高斯的像素投影弱标签、3D ID损失配置
                gs.means, self.classifier(gs.extras["id_features"]),
                gs.extras["instance_ids"], gs.opacities.reshape(-1),
                self.labels_at_points(gs.means, labels, camera), cfg3d)# 3d id loss：所有配对的锚点和候选id概率分布的平均KL散度
            losses["id_3d_loss"] = cfg3d.w * loss3d
            stats.update(stats3d)
        return losses, stats
