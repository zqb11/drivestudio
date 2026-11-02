import math
import unittest
from importlib import import_module

import torch
from torch.nn import functional as F

project_instance_labels = import_module("models.2d_id_loss").project_instance_labels
masked_id_cross_entropy = import_module("models.2d_id_loss").masked_id_cross_entropy


class InstanceProjectionTest(unittest.TestCase):
    def setUp(self):
        self.dynamic = torch.ones(5, 5)
        self.camera = torch.eye(4)
        self.intrinsics = torch.tensor([[5., 0., 2.5], [0., 5., 2.5], [0., 0., 1.]])
        self.pose = torch.eye(4).unsqueeze(0)
        self.pose[0, 2, 3] = 5.
        self.size = torch.tensor([[2., 2., 2.]])

    def project(self, poses=None, sizes=None, valid=None, **kwargs):
        poses = self.pose if poses is None else poses
        sizes = self.size if sizes is None else sizes
        valid = torch.ones(poses.shape[0], dtype=torch.bool) if valid is None else valid
        return project_instance_labels(
            kwargs.pop("dynamic_mask", self.dynamic),
            kwargs.pop("sky_mask", None), kwargs.pop("egocar_mask", None),
            kwargs.pop("camtoworld", self.camera),
            kwargs.pop("intrinsics", self.intrinsics), poses, sizes, valid, **kwargs,
        )

    def test_center_rays_and_pixel_mask(self):
        labels = self.project(chunk_size=2)
        self.assertTrue(torch.equal(labels[1:4, 1:4], torch.ones(3, 3, dtype=torch.long)))
        self.assertEqual(labels[0, 0].item(), -1)
        dynamic = self.dynamic.clone()
        dynamic[2, 2] = 0
        self.assertEqual(self.project(dynamic_mask=dynamic)[2, 2].item(), 0)

    def test_world_translation_and_rotation_do_not_change_labels(self):
        transform = torch.tensor([
            [0., 0., 1., 7.], [0., 1., 0., -2.],
            [-1., 0., 0., 3.], [0., 0., 0., 1.],
        ])
        moved = self.project(poses=transform.unsqueeze(0) @ self.pose, camtoworld=transform)
        self.assertTrue(torch.equal(moved, self.project()))

    def test_box_rotation_uses_oriented_extent(self):
        sizes = torch.tensor([[4., 1., 2.]])
        unrotated = self.project(sizes=sizes)
        pose = self.pose.clone()
        pose[0, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        rotated = self.project(poses=pose, sizes=sizes)
        self.assertTrue(torch.equal(rotated, unrotated.T))
        self.assertEqual(unrotated[2, 0].item(), 1)
        self.assertEqual(rotated[2, 0].item(), -1)

    def test_overlap_is_ignored_and_indices_are_stable(self):
        poses = self.pose.expand(2, -1, -1).clone()
        sizes = self.size.expand(2, -1).clone()
        self.assertEqual(self.project(poses, sizes)[2, 2].item(), -1)
        labels = self.project(poses, sizes, torch.tensor([False, True]))
        self.assertEqual(labels[2, 2].item(), 2)

    def test_empty_boxes_sky_and_ego(self):
        dynamic = torch.zeros(5, 5)
        dynamic[2, 2] = 1
        sky = torch.zeros(5, 5)
        ego = sky.clone()
        sky[0, 0] = 1
        ego[1, 1] = 1
        labels = self.project(
            poses=torch.empty(0, 4, 4), sizes=torch.empty(0, 3),
            dynamic_mask=dynamic, sky_mask=sky, egocar_mask=ego,
        )
        self.assertEqual(labels[2, 2].item(), -1)
        self.assertEqual(labels[0, 0].item(), -1)
        self.assertEqual(labels[1, 1].item(), -1)
        self.assertEqual(labels[4, 4].item(), 0)

    def test_near_plane_crossing_and_behind_boxes_are_ignored(self):
        poses = self.pose.clone()
        poses[0, 2, 3] = 0.5
        self.assertTrue((self.project(poses=poses) == -1).all())
        poses[0, 2, 3] = -5.
        self.assertTrue((self.project(poses=poses) == -1).all())

    def test_scaled_intrinsics_and_detached_labels(self):
        poses = self.pose.clone().requires_grad_()
        low = project_instance_labels(
            torch.ones(1, 1), None, None, self.camera,
            torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]),
            poses, self.size, torch.ones(1, dtype=torch.bool),
        )
        self.assertEqual(low.item(), 1)
        self.assertFalse(low.requires_grad)


class IdentityLossTest(unittest.TestCase):
    def test_mask_normalization_and_gradients(self):
        features = torch.randn(2, 2, 4, requires_grad=True)
        classifier = torch.nn.Linear(4, 3)
        logits = classifier(features)
        labels = torch.tensor([[0, 1], [-1, 2]])
        opacity = torch.tensor([[1., 0.01], [1., 0.5]], requires_grad=True)
        loss = masked_id_cross_entropy(logits, labels, opacity)
        expected = F.cross_entropy(logits[[0, 1], [0, 1]], torch.tensor([0, 2])) / math.log(3)
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertGreater(features.grad[0, 0].abs().sum().item(), 0)
        self.assertGreater(features.grad[1, 1].abs().sum().item(), 0)
        self.assertEqual(features.grad[0, 1].abs().sum().item(), 0)
        self.assertEqual(features.grad[1, 0].abs().sum().item(), 0)
        self.assertIsNone(opacity.grad)
        self.assertGreater(classifier.weight.grad.abs().sum().item(), 0)

    def test_empty_selection_is_differentiable_zero(self):
        logits = torch.randn(2, 2, 3, requires_grad=True)
        loss = masked_id_cross_entropy(logits, torch.full((2, 2), -1), torch.ones(2, 2, 1))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_nonfinite_and_low_opacity_are_ignored(self):
        logits = torch.zeros(1, 3, 2, requires_grad=True)
        loss = masked_id_cross_entropy(
            logits, torch.zeros(1, 3, dtype=torch.long), torch.tensor([[float("nan"), 0., 0.05]])
        )
        self.assertAlmostEqual(loss.item(), 1.)
        loss.backward()
        self.assertEqual(logits.grad[0, :2].abs().sum().item(), 0)

    def test_out_of_range_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            masked_id_cross_entropy(torch.zeros(1, 1, 2), torch.tensor([[2]]), torch.ones(1, 1))


if __name__ == "__main__":
    unittest.main()
