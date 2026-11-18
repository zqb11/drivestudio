"""Full-sequence, reversible cleanup of background Gaussians on moving instances.

ID learning is adapted from DIAL-GS; covariance projection follows 3DGS.
Audit scheduling, conditional depth completion, boundary protection, opacity
recovery and cleanup thresholds are project designs, not paper reproductions.
"""
import json
import logging
import os

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger()


class BackgroundCleanup(nn.Module):
    point_fields = (
        "instance", "support_count", "static_count", "camera_bits",
        "support_bits", "static_bits", "checked_bits", "observed_bits", "unknown_bits", "conflict_bits",
        "checked_count", "observed_count", "unknown_count", "conflict_count",
        "confirmed_at", "confirmed_rounds", "before_clean_opacity", "monitor_mask", "monitor_initial",
    )
    evidence_kinds = ("support", "static", "checked", "observed", "unknown", "conflict")

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        for name in self.point_fields:
            dtype = torch.bool if name == "monitor_mask" else (
                torch.float32 if name in ("monitor_initial", "before_clean_opacity") else torch.long)
            self.register_buffer(name, torch.empty(0, device=device, dtype=dtype))
        for name in ("monitor_step", "monitor_count", "deleted_total", "last_alert", "audit_round"):
            self.register_buffer(name, torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer("audit_step", torch.full((), -1, device=device, dtype=torch.long))
        self.register_buffer("monitor_raw_sum", torch.zeros((), device=device))
        self.last_evidence = {}
        self.gradient_stats = {}

    def initialize(self, num_points, num_frames):
        for name in self.point_fields:
            shape = (num_points, (num_frames + 62) // 63) if name.endswith("_bits") and name != "camera_bits" else (num_points,)
            self._buffers[name] = self._buffers[name].new_zeros(shape)
        self.confirmed_at.fill_(-1)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + "free_bits" in state_dict:
            raise RuntimeError("Legacy residual-clean checkpoint uses free-space evidence. "
                               "Evaluate it with its saved code; start the revised experiment from step 0.")
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
        self.confirmed_at[old_count:] = -1
        self.confirmed_rounds[old_count:] = 0

    @torch.no_grad()
    def retain(self, deleted):
        if not self.instance.numel():
            return
        self.deleted_total.add_(((self.confirmed_at >= 0) & deleted).sum())
        for name in self.point_fields:
            self._buffers[name] = self._buffers[name][~deleted]

    def ramp(self, step):
        return max(0., min(1., (step - self.cfg.start_step) / self.cfg.loss_ramp_steps))

    def score(self):
        return (self.confirmed_at >= 0).float()

    def gate(self, step):
        age = (step - self.confirmed_at).clamp_min(0)
        return self.score() * (age.float() / self.cfg.gate_ramp_steps).clamp(max=1.) * self.cfg.max_gate

    def apply_gate(self, opacity, step):
        if not self.instance.numel():
            return opacity
        return opacity * (1. - self.gate(step)[:, None])

    def deletion_mask(self, step):
        return ((self.score() > 0) & (step - self.confirmed_at >= self.cfg.delete_after_steps)
                & (self.confirmed_rounds >= self.cfg.min_confirmed_rounds))

    def protected_mask(self, step):
        return (self.score() > 0) & ~self.deletion_mask(step)

    def should_audit(self, step, final=False):
        return step >= self.cfg.start_step and (
            not self.instance.numel() or int(self.audit_step) < self.cfg.start_step
            or (step - self.cfg.start_step) % self.cfg.audit_interval == 0
            or step in self.cfg.discovery_steps or final)

    @torch.no_grad()
    def _add_frame(self, ids, frame, kind):
        bits = getattr(self, kind + "_bits")
        word, shift = divmod(frame, 63)
        bit = 1 << shift
        fresh = (bits[ids, word] & bit) == 0
        bits[ids, word] |= bit
        getattr(self, kind + "_count")[ids[fresh]] += 1

    def _observation(self, image_set, index, identity, device):
        image, camera = image_set.get_image(index, camera_downscale=1.0)
        frame = int(image["frame_idx"].flatten()[0])
        image = {key: image[key].to(device) for key in (
            "dynamic_masks", "sky_masks", "egocar_masks", "cleanup_depth_map") if key in image}
        camera = {key: camera[key].to(device) for key in ("camera_to_world", "intrinsics", "cam_id")}
        labels = identity.image_labels(image, camera, frame)
        blocked = (labels != 0) | (image["dynamic_masks"] > .5)
        margin = self.cfg.boundary_margin_pixels
        safe = F.max_pool2d(blocked.float()[None, None], 2 * margin + 1, stride=1, padding=margin)[0, 0] == 0
        integral = F.pad(blocked.long().cumsum(0).cumsum(1), (1, 0, 1, 0))
        return frame, image, camera, labels, safe, integral

    def _sample_depth(self, depth, labels, uv, safe):
        """Keep valid measurements; fill only unknown pixels from coherent same-ID neighbors."""
        x, y = uv.unbind(-1)
        direct = depth[y, x]
        valid = torch.isfinite(direct) & (direct > 0)
        measured = torch.where(valid, direct, torch.full_like(direct, torch.inf))
        source = valid.long()
        missing = torch.where(~valid & (labels[y, x] >= 0))[0]
        mx, my = x[missing], y[missing]
        center_labels = labels[my, mx]
        h, w = depth.shape
        values, accepted = [], []
        radius = self.cfg.depth_neighborhood
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = mx + dx, my + dy
                inside = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
                nx, ny = nx.clamp(0, w - 1), ny.clamp(0, h - 1)
                value = depth[ny, nx]
                same_id = labels[ny, nx] == center_labels
                background_safe = (center_labels > 0) | (safe[my, mx] & safe[ny, nx])
                accepted.append(inside & same_id & background_safe & torch.isfinite(value) & (value > 0))
                values.append(value)
        values, accepted = torch.stack(values, -1), torch.stack(accepted, -1)
        count = accepted.sum(-1)
        sorted_values = values.masked_fill(~accepted, torch.inf).sort(-1).values
        low = sorted_values[:, 0]
        high = values.masked_fill(~accepted, -torch.inf).max(-1).values
        median = sorted_values.gather(1, ((count - 1).clamp_min(0) // 2)[:, None])[:, 0]
        coherent = ((count >= self.cfg.min_depth_neighbors)
                    & (high - low <= self.cfg.fill_depth_margin + self.cfg.fill_depth_relative_margin * median))
        filled_ids = missing[coherent]
        measured[filled_ids] = low[coherent]
        source[filled_ids] = 2
        return measured, source

    def _static_footprint_safe(self, model, ids, xyz, uv, camera, integral):
        """Project covariance with the original camera, avoiding dynamic/ignored regions."""
        from models.gaussians.basics import quat_to_rotmat
        rotation = quat_to_rotmat(model.get_quats[ids])
        scaled = rotation * model.get_scaling[ids, None, :]
        covariance = scaled @ scaled.transpose(-1, -2)
        camera_rotation = camera["camera_to_world"][:3, :3]
        covariance = camera_rotation.T @ covariance @ camera_rotation
        k = camera["intrinsics"]
        z = xyz[:, 2]
        jacobian = xyz.new_zeros((ids.numel(), 2, 3))
        jacobian[:, 0, 0] = k[0, 0] / z
        jacobian[:, 1, 1] = k[1, 1] / z
        jacobian[:, 0, 2] = -k[0, 0] * xyz[:, 0] / z.square()
        jacobian[:, 1, 2] = -k[1, 1] * xyz[:, 1] / z.square()
        projected = jacobian @ covariance @ jacobian.transpose(-1, -2)
        radius = (self.cfg.footprint_sigma * projected.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
                  + self.cfg.boundary_margin_pixels).ceil().long()
        h, w = integral.shape[0] - 1, integral.shape[1] - 1
        low, high = uv - radius, uv + radius + 1
        within_image = (low >= 0).all(-1) & (high[:, 0] <= w) & (high[:, 1] <= h)
        x0, y0 = low[:, 0].clamp(0, w), low[:, 1].clamp(0, h)
        x1, y1 = high[:, 0].clamp(0, w), high[:, 1].clamp(0, h)
        blocked = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
        return within_image & (blocked == 0)

    def _observe(self, model, identity, ids, frame, image, camera, labels, safe, integral, discover=False):
        means = model._means.detach()[ids]
        c2w = camera["camera_to_world"]
        xyz = (means - c2w[:3, 3]) @ c2w[:3, :3]
        projected = xyz @ camera["intrinsics"].T
        uv = projected[:, :2] / xyz[:, 2:].clamp_min(1e-6)
        h, w = labels.shape
        inside = ((xyz[:, 2] > .1) & torch.isfinite(uv).all(-1) & (uv[:, 0] >= 0)
                  & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        local_ids = torch.where(inside)[0]
        selected = ids[local_ids]
        xyz, uv, means = xyz[inside], uv[inside].long(), means[inside]
        pixel_labels = labels[uv[:, 1], uv[:, 0]]
        measured, source = self._sample_depth(image["cleanup_depth_map"], labels, uv, safe)
        observed = torch.isfinite(measured) & (pixel_labels >= 0)
        surface = observed & ((xyz[:, 2] - measured).abs()
                              <= self.cfg.surface_margin + self.cfg.surface_relative_margin * xyz[:, 2])
        dynamic = (pixel_labels > 0) & surface
        candidates = torch.where(dynamic)[0]
        if candidates.numel():
            candidate_labels = pixel_labels[candidates]
            poses = identity.poses[frame, candidate_labels - 1]
            local = torch.bmm((means[candidates] - poses[:, :3, 3])[:, None], poses[:, :3, :3]).squeeze(1)
            in_box = (local.abs() <= identity.sizes[candidate_labels - 1] / 2 + self.cfg.box_margin).all(-1)
            probabilities = identity.classifier(model._id_features[selected[candidates]]).softmax(-1)
            confident = probabilities.gather(1, candidate_labels[:, None])[:, 0] >= self.cfg.id_confidence
            associated = self.instance[selected[candidates]]
            same_instance = (associated == 0) | (associated == candidate_labels)
            reliable = in_box & confident & same_instance & identity.valid[frame, candidate_labels - 1]
            dynamic[candidates] = reliable
            if discover:
                self.instance[selected[candidates[reliable]]] = candidate_labels[reliable]
        if discover:
            return
        static = (pixel_labels == 0) & surface & safe[uv[:, 1], uv[:, 0]]
        static_candidates = torch.where(static)[0]
        if static_candidates.numel():
            static[static_candidates] &= self._static_footprint_safe(
                model, selected[static_candidates], xyz[static_candidates], uv[static_candidates], camera, integral)
        unknown = ~observed | ((pixel_labels == 0) & surface & ~static)
        result = []
        for mask in (dynamic, static, observed, unknown):
            full = torch.zeros(ids.numel(), dtype=torch.bool, device=ids.device)
            full[local_ids] = mask
            result.append(full)
        for name, mask in (("depth_direct", source == 1), ("depth_filled", source == 2),
                           ("depth_unknown", source == 0), ("depth_conflict", observed & ~surface)):
            self.last_evidence[name] += int(mask.sum())
        return result

    @torch.no_grad()
    def audit(self, model, identity, image_set, step, optimizer, discover=False, log_dir=None):
        """Hold parameters fixed while discovering and auditing all training observations."""
        step = int(step)
        if not self.instance.numel():
            self.initialize(model.num_points, identity.poses.shape[0])
            discover = True
        device, chunk = model._means.device, self.cfg.max_points_per_update
        logger.info("Background cleanup audit at %d: discovery=%s, training images=%d", step, discover, len(image_set))
        if discover:
            unassociated = torch.where(self.instance == 0)[0]
            for index in range(len(image_set)):
                observation = self._observation(image_set, index, identity, device)
                for start in range(0, unassociated.numel(), chunk):
                    ids = unassociated[start:start + chunk]
                    ids = ids[self.instance[ids] == 0]
                    if ids.numel():
                        self._observe(model, identity, ids, *observation, discover=True)
                if (index + 1) % 50 == 0:
                    logger.info("Background cleanup discovery: %d/%d images", index + 1, len(image_set))
        tracked = torch.where(self.instance > 0)[0]
        for kind in self.evidence_kinds:
            getattr(self, kind + "_bits").zero_()
            getattr(self, kind + "_count").zero_()
        self.camera_bits.zero_()
        self.last_evidence = dict(depth_direct=0, depth_filled=0, depth_unknown=0, depth_conflict=0)
        frames = {}
        for index, image_index in enumerate(image_set.split_indices):
            frame = image_set.datasource.parse_img_idx(int(image_index))[1]
            frames.setdefault(frame, []).append(index)
        for frame in sorted(frames):
            dynamic = torch.zeros(tracked.numel(), dtype=torch.bool, device=device)
            static, observed, unknown = torch.zeros_like(dynamic), torch.zeros_like(dynamic), torch.zeros_like(dynamic)
            cameras = torch.zeros(tracked.numel(), dtype=torch.long, device=device)
            if tracked.numel():
                for index in frames[frame]:
                    observation = self._observation(image_set, index, identity, device)
                    observed_frame, _, camera, _, _, _ = observation
                    if observed_frame != frame:
                        raise ValueError("Cleanup audit image/frame mapping mismatch")
                    camera_bit = 1 << int(camera["cam_id"].flatten()[0])
                    for start in range(0, tracked.numel(), chunk):
                        part = slice(start, start + chunk)
                        d, s, o, u = self._observe(model, identity, tracked[part], *observation)
                        dynamic[part] |= d
                        static[part] |= s
                        observed[part] |= o
                        unknown[part] |= u
                        cameras[part] |= d.long() * camera_bit
            conflict = dynamic & static
            dynamic &= ~conflict
            static &= ~conflict
            self._add_frame(tracked, frame, "checked")
            for kind, mask in (("support", dynamic), ("static", static), ("observed", observed),
                               ("unknown", unknown & ~(dynamic | static)), ("conflict", conflict)):
                self._add_frame(tracked[mask], frame, kind)
            self.camera_bits[tracked] |= cameras * (~conflict).long()
        camera_count = torch.zeros_like(self.camera_bits)
        for camera in image_set.datasource.camera_list:
            camera_count += (self.camera_bits >> int(camera)) & 1
        confirmed = ((self.instance > 0) & (self.support_count >= self.cfg.min_support_frames)
                     & (camera_count >= self.cfg.min_cameras)
                     & (self.static_count < self.cfg.static_veto_frames)
                     & (self.checked_count == len(frames)))
        previous = self.confirmed_at >= 0
        revoked, new = previous & ~confirmed, confirmed & ~previous
        model._opacities[revoked, 0] = self.before_clean_opacity[revoked]
        state = optimizer.state.get(model._opacities, {})
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if name in state:
                state[name][revoked] = 0
        self.before_clean_opacity[new] = model._opacities[new, 0]
        self.confirmed_at[~confirmed] = -1
        self.confirmed_at[new] = step
        self.confirmed_rounds[confirmed] += 1
        self.confirmed_rounds[~confirmed] = 0
        self.audit_step.fill_(step)
        self.audit_round.add_(1)
        self.gradient_stats = {}
        self.last_evidence.update(audit_step=step, audit_round=int(self.audit_round), audit_images=len(image_set),
                                  audit_frames=len(frames), discovered_candidates=int(tracked.numel()),
                                  confirmed_new=int(new.sum()), confirmed_revoked=int(revoked.sum()),
                                  supported_frames_ready=int((self.support_count >= self.cfg.min_support_frames).sum()),
                                  insufficient_dynamic_support=int((self.support_count[tracked] < self.cfg.min_support_frames).sum()),
                                  fully_checked=int((self.checked_count[tracked] == len(frames)).sum()),
                                  checked_frames_total=int(self.checked_count[tracked].sum()),
                                  observed_frames_total=int(self.observed_count[tracked].sum()),
                                  unknown_frames_total=int(self.unknown_count[tracked].sum()),
                                  conflict_frames_total=int(self.conflict_count[tracked].sum()))
        logger.info("Background cleanup audit complete: %s", json.dumps(self.last_evidence))
        if log_dir is not None:
            with open(os.path.join(log_dir, "cleanup_audits.jsonl"), "a") as stream:
                stream.write(json.dumps(self.last_evidence) + "\n")

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
