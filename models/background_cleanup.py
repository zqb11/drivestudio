"""Evidence-based removal of static Gaussians left on moving vehicles."""
import json
import logging
import os

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger()

# 背景残影高斯清理模块
class BackgroundCleanup(nn.Module):
    point_fields = (
        "instance", "support_count", "free_count", "static_count", "camera_bits",
        "support_bits", "free_bits", "static_bits", "confirmed_at", "monitor_mask", "monitor_initial",
    )

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        for name in self.point_fields:
            dtype = torch.bool if name == "monitor_mask" else (torch.float32 if name == "monitor_initial" else torch.long)
            self.register_buffer(name, torch.empty(0, device=device, dtype=dtype))
        for name in ("monitor_step", "monitor_count", "deleted_total", "cursor", "last_alert"):
            self.register_buffer(name, torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer("monitor_raw_sum", torch.zeros((), device=device))
        self.last_evidence = {}
        self.gradient_stats = {}

    def initialize(self, num_points, num_frames):
        for name in self.point_fields:
            shape = (num_points, (num_frames + 62) // 63) if name.endswith("_bits") and name != "camera_bits" else (num_points,)# 支持帧记录使用的整数个数
            self._buffers[name] = self._buffers[name].new_zeros(shape)
        self.confirmed_at.fill_(-1)# 每个背景高斯首次被确认是残影、需要清理的训练步数，初始化为-1

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for name in self.point_fields:
            self._buffers[name] = torch.empty_like(state_dict[prefix + name], device=self.monitor_step.device)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def append(self, splits, duplicates, n):
        if not self.instance.numel():
            return
        old_count = self.instance.shape[0]
        for name in self.point_fields:
            values = self._buffers[name]
            repeats = (n,) + (1,) * (values.ndim - 1)
            self._buffers[name] = torch.cat((values, values[splits].repeat(repeats), values[duplicates]))
        self.monitor_mask[old_count:] = False
    
    @torch.no_grad()
    def retain(self, deleted):
        if not self.instance.numel():
            return
        self.deleted_total.add_(((self.confirmed_at >= 0) & deleted).sum())# 累计“曾被确认为残影、且本次被裁剪”的高斯数量
        for name in self.point_fields:# 遍历清理模块的属性
            self._buffers[name] = self._buffers[name][~deleted]# 删除属性中被裁剪的高斯对应行

    def ramp(self, step):
        return max(0., min(1., (step - self.cfg.start_step) / self.cfg.loss_ramp_steps))
    # 判断背景高斯是否是节点残影
    def score(self):
        return ((self.confirmed_at >= 0) & (self.static_count < self.cfg.static_veto_frames)).float()# 背景残影高斯掩码：曾经被确认是残影且当前静态支持帧数小于阈值的高斯为1，否则为0
    # 门控强度
    def gate(self, step):
        age = (step - self.confirmed_at).clamp_min(0)# 每个高斯首次确认是残影后经过了多少步，表示残影高斯的年龄
        return self.score() * (age.float() / self.cfg.gate_ramp_steps).clamp(max=1.) * self.cfg.max_gate# 门控强度：随着残影高斯的年龄越来越大，门控强度越来越大，直到1
    # 
    def apply_gate(self, opacity, step):
        if not self.instance.numel():
            return opacity
        return opacity * (1. - self.gate(step)[:, None])# 用门控强度逐渐缩小残影高斯的不透明度，让其逐渐消失
    # 背景残影高斯掩码
    def deletion_mask(self, step):
        return (self.score() > 0) & (step - self.confirmed_at >= self.cfg.delete_after_steps)# 当前确认是残影且年龄大于阈值

    @torch.no_grad()
    def _add_frame(self, ids, frame, kind):
        bits = getattr(self, kind + "_bits")# 背景高斯的静态/自由/动态支持帧掩码（几个63位整数，分别记录某时间步是否支持，支持为1，不支持为0）
        word, shift = divmod(frame, 63)# 当前时间步在第几个整数中，以及在该整数的第几位，编号都从0开始
        bit = 1 << shift
        fresh = (bits[ids, word] & bit) == 0# 是否得到该时间步的第一次静态/自由/动态支持，true表示该时间步是第一次支持，false表示该时间步已经支持过
        bits[ids, word] |= bit# 将背景高斯的静态/自由/动态支持帧掩码中该时间步的位设置为1，表示该时间步已经支持过
        getattr(self, kind + "_count")[ids[fresh]] += 1# 背景高斯的静态/自由/动态支持帧数：只有在该时间步第一次支持时，才增加支持帧数
    # 更新背景高斯的静态/自由/动态支持时间帧掩码和支持帧数、动态支持相机掩码和相机数，得到背景高斯首次被确认是残影的训练步数
    @torch.no_grad()
    def update(self, model, identity, labels, image_infos, cam_infos, frame, step):
        if step < self.cfg.start_step or step % self.cfg.evidence_interval:# 从训练25000步开始更新，更新间隔是10
            return
        if not self.instance.numel():
            self.initialize(model.num_points, identity.poses.shape[0])# 初始化背景残影高斯首次被确认的训练步数
        count = min(model.num_points, self.cfg.max_points_per_update)# 确定本次扫描多少个高斯
        ids = (torch.arange(count, device=model._means.device) + self.cursor) % model.num_points# 本次扫描高斯编号：从上次扫描结束的位置开始，生成本次扫描编号，如果超过总编号，则从0开始
        self.cursor.copy_((self.cursor + count) % model.num_points)# 保存下一次扫描的起点
        tracked = torch.where((self.instance > 0) & (self.static_count < self.cfg.static_veto_frames))[0]# 跟踪高斯编号：所有已关联车辆节点、且未被静态证据否决的高斯编号
        ids = torch.cat((ids[self.instance[ids] == 0], tracked))# 合并本次扫描中未关联节点的高斯编号和已跟踪高斯编号
        means = model._means.detach()[ids]
        c2w = cam_infos["camera_to_world"]
        xyz = (means - c2w[:3, 3]) @ c2w[:3, :3]
        projected = xyz @ cam_infos["intrinsics"].T
        uv = projected[:, :2] / xyz[:, 2:].clamp_min(1e-6)# 未关联节点和已跟踪高斯中心投影到当前图像
        h, w = labels.shape
        inside = (xyz[:, 2] > .1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids, xyz, uv, means = ids[inside], xyz[inside], uv[inside].long(), means[inside]# 在相机视野且在相机前方的未关联节点和已跟踪高斯编号、相机中心坐标、投影坐标和世界中心坐标
        pixel_labels = labels[uv[:, 1], uv[:, 0]]# 未关联节点和已跟踪高斯的投影像素点的弱标签
        depth = image_infos["cleanup_depth_map"]
        radius = self.cfg.depth_neighborhood
        filled = depth.masked_fill(depth <= 0, torch.inf)# 清理深度图<=0替换为正无穷
        nearest = -F.max_pool2d(-filled[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]# 最小池化的清理深度图：清理深度图通过“取负 → 最大池化 → 再取负”实现最小池化，每个像素取以它为中心的3×3区域内的最小深度
        measured = nearest[uv[:, 1], uv[:, 0]]# 未关联节点和已跟踪高斯的目标深度（LiDAR 观测点的相机z深度）
        observed = torch.isfinite(measured)# 已知的目标深度
        delta = xyz[:, 2] - measured# 未关联节点和已跟踪高斯深度与目标深度之差
        surface_margin = self.cfg.surface_margin + self.cfg.surface_relative_margin * xyz[:, 2]# 高斯深度与目标深度误差=固定容差+随深度增加的容差
        surface = observed & (delta.abs() <= surface_margin)# 表面附近的高斯：未关联节点和已跟踪高斯深度接近目标深度且目标深度已知时
        free = observed & (-delta > self.cfg.free_margin + self.cfg.free_relative_margin * xyz[:, 2])# 自由空间高斯/位于表面前方高斯：未关联节点和已跟踪高斯深度明显小于目标深度且目标深度已知时
        # 静态支持和自由空间支持，更新静态/自由支持时间帧掩码和支持帧数
        static_ids = ids[(pixel_labels == 0) & surface]# 静态支持高斯编号：投影区域是背景且位于背景表面附近。 
        free_ids = ids[(pixel_labels == 0) & free]# 自由空间支持高斯编号：投影区域是背景且位于背景表面前方。 
        # 如果该时间步是第一次支持，就把静态/自由支持时间帧掩码的该时间步的位设置为1，并加到静态/自由支持帧数
        self._add_frame(static_ids, frame, "static")
        self._add_frame(free_ids, frame, "free")
        # 筛选可靠的动态支持
        dynamic = (pixel_labels > 0) & surface# 动态支持：投影区域是节点且位于车辆表面附近
        candidate_ids, candidate_labels = ids[dynamic], pixel_labels[dynamic]# 动态支持高斯编号和动态支持高斯弱标签
        candidate_means = means[dynamic]# 动态支持高斯的世界中心坐标
        self.last_evidence = {"surface_dynamic": candidate_ids.numel(), "reliable_dynamic": 0,
                              "free_observations": free_ids.numel(), "static_observations": static_ids.numel()}
        if candidate_ids.numel():
            poses = identity.poses[frame, candidate_labels - 1]
            local = torch.bmm((candidate_means - poses[:, :3, 3])[:, None], poses[:, :3, :3]).squeeze(1)# 将动态支持高斯的世界中心坐标转换为局部坐标
            in_box = (local.abs() <= identity.sizes[candidate_labels - 1] / 2 + self.cfg.box_margin).all(-1)# 条件一：检查中心是否位于扩大 0.2m 的实例框内，约束高斯的三维位置
            probabilities = identity.classifier(model._id_features[candidate_ids]).softmax(-1)
            #agrees = probabilities.argmax(-1) == candidate_labels# 预测ID与像素投影弱标签是否一致
            confident = probabilities.gather(1, candidate_labels[:, None])[:, 0] >= self.cfg.id_confidence# 条件二：预测id取像素标签的概率至少为0.6，约束预测id与像素标签一致
            same_instance = (self.instance[candidate_ids] == 0) | (self.instance[candidate_ids] == candidate_labels)# 条件三：尚未关联节点，或此前关联的就是这个节点，约束高斯始终关联同一个节点
            # reliable = in_box & agrees & confident & same_instance# 中心在实例框内、预测ID与像素投影弱标签一致、预测ID 的概率至少为 0.6、尚未关联实例，或此前关联的就是这个实例
            reliable = in_box & confident & same_instance# 条件一、条件二、条件三
            reliable_ids, reliable_labels = candidate_ids[reliable], candidate_labels[reliable]# 动态支持高斯经过三个条件筛选后得到残影候选高斯编号和像素标签
            self.instance[reliable_ids] = reliable_labels# 残影候选高斯的关联节点
            self._add_frame(reliable_ids, frame, "support")# 更新动态支持时间帧掩码和支持帧数
            cam_id = int(cam_infos["cam_id"].flatten()[0])# 目标图像的相机视角
            self.camera_bits[reliable_ids] |= 1 << cam_id# 更新动态支持相机掩码：一个二进制整数，每一位表示一个相机，1表示该高斯在该相机视野内被动态支持过，0表示没有
            self.last_evidence = {"surface_dynamic": candidate_ids.numel(), "reliable_dynamic": reliable_ids.numel(),
                                  "free_observations": free_ids.numel(), "static_observations": static_ids.numel()}
        camera_count = torch.zeros_like(self.camera_bits)# 更新动态支持相机数
        for cam in range(6):
            camera_count += (self.camera_bits >> cam) & 1
        confirmed = ((self.support_count >= self.cfg.min_support_frames)
                     & (self.free_count >= self.cfg.min_free_frames)
                     & (camera_count >= self.cfg.min_cameras)
                     & (self.static_count < self.cfg.static_veto_frames))# 已确认残影高斯：动态支持帧数、自由支持帧数和动态支持相机数至少达到阈值，且静态支持帧数小于静态否决帧数
        self.confirmed_at[confirmed & (self.confirmed_at < 0)] = step# 背景高斯首次被确认是残影的训练步数：只有在该高斯首次被确认是残影时，才记录训练步数

    def loss(self, model, step):
        if not self.instance.numel():
            return model._opacities.sum() * 0.
        selected = self.score() > 0# 施加清理损失的残影高斯集合
        return self.cfg.w * self.ramp(step) * F.softplus(model._opacities[selected]).sum() / selected.sum().clamp_min(1)

    @torch.no_grad()
    def capture_gradient(self, model):
        if not self.instance.numel() or model._opacities.grad is None:
            return
        selected = self.score() > 0
        if selected.any():
            grad = model._opacities.grad[selected]
            self.gradient_stats = {"opacity_logit_gradient_mean": grad.mean().item(),
                                   "opacity_decrease_gradient_fraction": (grad > 0).float().mean().item()}
    # 清理效果监测：
    @torch.no_grad()
    def report(self, model, step, log_dir):
        step = int(step)
        if step < self.cfg.start_step or step % self.cfg.report_interval:
            return
        opacity = model.get_opacity.detach().flatten()
        selected = self.score().bool() if self.instance.numel() else torch.zeros_like(opacity, dtype=torch.bool)
        n = int(selected.sum())
        stats = {"step": step, "confirmed": n, "deleted_total": int(self.deleted_total),
                 "loss_weight": self.cfg.w * self.ramp(step), "gate_strength": self.gate(step)[selected].mean().item() if n else 0.,
                 "raw_opacity_mean": opacity[selected].mean().item() if n else None,
                 "effective_opacity_mean": self.apply_gate(opacity[:, None], step).flatten()[selected].mean().item() if n else None,
                 "opacity_reset_step": int(step % model.ctrl_cfg.reset_alpha_interval == model.ctrl_cfg.refine_interval)}
        stats.update(self.last_evidence)
        stats.update(self.gradient_stats)
        if self.instance.numel():
            stats.update(tracked_with_id=int((self.instance > 0).sum()),
                         supported_frames_ready=int((self.support_count >= self.cfg.min_support_frames).sum()),
                         free_frames_ready=int(((self.instance > 0) & (self.free_count >= self.cfg.min_free_frames)).sum()),
                         static_vetoed=int(((self.instance > 0) & (self.static_count >= self.cfg.static_veto_frames)).sum()))
        alerts = []
        if self.monitor_count > 0:
            revoked = self.monitor_mask & ~selected
            self.monitor_count.sub_(revoked.sum())
            self.monitor_raw_sum.sub_(self.monitor_initial[revoked].sum())
            self.monitor_mask[revoked] = False
        if self.monitor_count > 0 and step - int(self.monitor_step) >= self.cfg.monitor_interval:
            initial = self.monitor_raw_sum.item() / int(self.monitor_count)
            current = opacity[self.monitor_mask].sum().item() / int(self.monitor_count)
            stats.update(cohort_initial_raw=initial, cohort_current_raw=current,
                         cohort_drop_fraction=1 - current / max(initial, 1e-8), cohort_size=int(self.monitor_count),
                         cohort_includes_opacity_reset=any(s % model.ctrl_cfg.reset_alpha_interval == model.ctrl_cfg.refine_interval
                                                          for s in range(int(self.monitor_step) + 1, step + 1)))
            survivors = int(self.monitor_mask.sum())
            if survivors:
                survivor_initial = self.monitor_initial[self.monitor_mask].mean().item()
                survivor_current = opacity[self.monitor_mask].mean().item()
                stats.update(cohort_survivors=survivors, survivor_initial_raw=survivor_initial,
                             survivor_current_raw=survivor_current,
                             survivor_raw_drop_fraction=1 - survivor_current / max(survivor_initial, 1e-8))
                if survivor_current >= survivor_initial * (1. - self.cfg.min_raw_drop_fraction):# 原始不透明度均值没有下降超过 1%，触发警告
                    alerts.append("RAW_OPACITY_NOT_DECREASING: same surviving Gaussians have not reduced raw opacity by the required fraction; gating/deletion alone is not opacity optimization")
            self.monitor_count.zero_()
        if self.monitor_count == 0 and n:
            self.monitor_mask.copy_(selected)
            self.monitor_step.fill_(step)
            self.monitor_count.fill_(n)
            self.monitor_raw_sum.copy_(opacity[selected].sum())
            self.monitor_initial.copy_(opacity)
        if not n and self.deleted_total == 0 and step - self.cfg.start_step >= self.cfg.no_confirmation_steps and step - int(self.last_alert) >= self.cfg.no_confirmation_steps:# 从清理起始步经过 2000 步后，如果当前没有残影高斯、历史上也没有删除过残影高斯，则触发
            alerts.append("NO_CONFIRMED_RESIDUALS: insufficient identity/depth/multi-frame evidence; cleanup is not yet active")
            self.last_alert.fill_(step)
        with open(os.path.join(log_dir, "cleanup_metrics.jsonl"), "a") as stream:
            stream.write(json.dumps(stats) + "\n")
        logger.info("Background cleanup: %s", json.dumps(stats))
        for message in alerts:
            alert = dict(stats, alert=message)
            with open(os.path.join(log_dir, "cleanup_alerts.jsonl"), "a") as stream:
                stream.write(json.dumps(alert) + "\n")
            logger.warning("Background cleanup alert: %s", json.dumps(alert))
