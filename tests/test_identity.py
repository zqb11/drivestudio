"""CPU integration checks for the scene-wide identity field."""

import copy
import unittest
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from models.identity import IdentityField, id_3d_enabled


class IdentityFieldTest(unittest.TestCase):
    def make_field(self):
        poses = torch.eye(4).repeat(2, 3, 1, 1)
        poses[:, :, 2, 3] = 5.
        poses[:, 1:, 0, 3] = 50.
        poses[1, 1, 0, 3] = 0.
        valid = torch.ones((2, 3), dtype=torch.bool)
        valid[1, 0] = False
        field = IdentityField(
            4, {"RigidNodes": [7, 2], "DeformableNodes": [9]},
            poses, torch.full((3, 3), 2.), valid, torch.tensor([70, 20, 90]),
        )
        with torch.no_grad():
            field.classifier.weight.copy_(torch.eye(4))
            field.classifier.bias.zero_()
        return field

    def make_images(self):
        dynamic = torch.zeros((5, 5))
        dynamic[2, 2] = 1.
        intrinsics = torch.tensor([[5., 0., 2.5], [0., 5., 2.5], [0., 0., 1.]])
        return {"dynamic_masks": dynamic}, {
            "camera_to_world": torch.eye(4), "intrinsics": intrinsics,
        }

    def test_3d_schedule_includes_25000_and_respects_weight(self):
        for step, expected in ((0, False), (24999, False), (25000, True), (25001, True)):
            with self.subTest(step=step):
                self.assertEqual(id_3d_enabled(step, {"w": 1.5}), expected)
        self.assertFalse(id_3d_enabled(25000, {"w": 0.}))
        self.assertFalse(id_3d_enabled(25000, {"w": -1.}))
        self.assertFalse(id_3d_enabled(25000, {"w": 1., "start_step": 25001}))
        self.assertTrue(id_3d_enabled(25001, {"w": 1., "start_step": 25001}))

    def test_checkpoint_reconstruction_restores_classifier_buffers_and_mapping(self):
        original = nn.Module()
        original.identity_field = self.make_field()
        state = copy.deepcopy(original.state_dict())
        restored = nn.Module()
        restored.identity_field = IdentityField.from_checkpoint(state)
        incompatible = restored.load_state_dict(state, strict=True)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(restored.identity_field.get_extra_state(), state["identity_field._extra_state"])
        for key, expected in state.items():
            actual = restored.state_dict()[key]
            if torch.is_tensor(expected):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            else:
                self.assertEqual(actual, expected)
        features = torch.tensor([[0., 6., 1., 2.], [4., 2., 1., 0.]])
        torch.testing.assert_close(
            restored.identity_field.classifier(features),
            original.identity_field.classifier(features), rtol=0, atol=0,
        )
        local_state = copy.deepcopy(original.identity_field.state_dict())
        local_restored = IdentityField.from_checkpoint(local_state, prefix="")
        local_restored.load_state_dict(local_state)
        self.assertEqual(local_restored.node_dataset_ids, {"RigidNodes": [7, 2], "DeformableNodes": [9]})

    def test_point_labels_use_global_offsets_and_local_instance_indices(self):
        field = self.make_field()
        model = SimpleNamespace(
            num_points=3, _means=torch.zeros((3, 3)), point_ids=torch.tensor([[1], [0], [1]]),
        )
        torch.testing.assert_close(field.point_labels("Background", model), torch.tensor([0, 0, 0]))
        torch.testing.assert_close(field.point_labels("RigidNodes", model), torch.tensor([2, 1, 2]))
        model.point_ids = torch.zeros((3, 1), dtype=torch.long)
        torch.testing.assert_close(field.point_labels("DeformableNodes", model), torch.tensor([3, 3, 3]))

    def test_duplicate_instance_ownership_is_rejected(self):
        field = self.make_field()
        with self.assertRaisesRegex(ValueError, "cannot belong to two"):
            IdentityField(4, {"RigidNodes": [7], "DeformableNodes": [7]},
                          field.poses[:, :2], field.sizes[:2], field.valid[:, :2], field.true_ids[:2])

    def test_multiframe_labels_keep_instance_identity_when_validity_changes(self):
        field = self.make_field()
        images, cameras = self.make_images()
        first = field.image_labels(images, cameras, frame=0)
        second = field.image_labels(images, cameras, frame=1)
        self.assertEqual(first[2, 2].item(), 1)
        self.assertEqual(second[2, 2].item(), 2)
        self.assertEqual((first > 0).sum().item(), 1)
        self.assertEqual((second > 0).sum().item(), 1)
        self.assertTrue((first[images["dynamic_masks"] == 0] == 0).all())
        self.assertFalse(first.requires_grad)
        self.assertFalse(second.requires_grad)

    def test_labels_at_points_use_processed_camera_and_ignore_hidden_points(self):
        field = self.make_field()
        images, cameras = self.make_images()
        labels = field.image_labels(images, cameras, frame=0)
        processed_pose = cameras["camera_to_world"].clone()
        processed_pose[0, 3] = 10.
        camera = SimpleNamespace(camtoworlds=processed_pose, Ks=cameras["intrinsics"])
        means = torch.tensor([
            [10., 0., 5.], [10.1, 0., 5.], [10., 0., -5.],
            [100., 0., 5.], [10., 0., .05], [float("nan"), 0., 5.],
        ], requires_grad=True)
        projected = field.labels_at_points(means, labels, camera)
        torch.testing.assert_close(projected, torch.tensor([1, 1, -1, -1, -1, -1]))
        self.assertFalse(projected.requires_grad)

    def test_compute_losses_24999_and_25000_values_keys_and_gradients(self):
        for step in (24999, 25000):
            with self.subTest(step=step):
                field = self.make_field()
                images, cameras = self.make_images()
                image_features = torch.zeros((5, 5, 4), requires_grad=True)
                point_features = torch.tensor([[0., 6., 0., 0.], [5., 0., 0., 0.]], requires_grad=True)
                gaussians = SimpleNamespace(
                    means=torch.tensor([[0., 0., 5.], [.1, 0., 5.]]),
                    opacities=torch.ones((2, 1)),
                    extras={"id_features": point_features, "instance_ids": torch.tensor([1, 0])},
                )
                outputs = {
                    "id_features": image_features, "opacity": torch.ones((5, 5, 1)),
                    "id_gaussians": gaussians,
                    "id_camera": SimpleNamespace(camtoworlds=torch.eye(4), Ks=cameras["intrinsics"]),
                }
                cfg = OmegaConf.create({
                    "id_2d": {"w": .1, "min_opacity": .05},
                    "id_3d": {"w": 1.5, "start_step": 25000, "confidence": .8, "max_distance": .5},
                })
                losses, stats = field.compute_losses(outputs, images, cameras, 0, step, cfg)
                expected_keys = {"id_2d_loss"} if step == 24999 else {"id_2d_loss", "id_3d_loss"}
                self.assertEqual(set(losses), expected_keys)
                self.assertAlmostEqual(losses["id_2d_loss"].item(), .1, places=6)
                self.assertEqual(stats["valid_pixels"], 25)
                self.assertEqual(stats["dynamic_pixels"], 1)
                self.assertEqual(stats["dynamic_id_accuracy"], 0.)
                self.assertEqual(outputs["id_labels"][2, 2].item(), 1)
                if step == 25000:
                    log_probabilities = F.log_softmax(point_features, dim=-1)
                    expected = 1.5 * (log_probabilities[0].exp() * (
                        log_probabilities[0] - log_probabilities[1]
                    )).sum() / 4
                    torch.testing.assert_close(losses["id_3d_loss"], expected)
                    self.assertEqual(stats["num_pairs"], 1)
                else:
                    self.assertNotIn("num_pairs", stats)
                sum(losses.values()).backward()
                self.assertGreater(image_features.grad.abs().sum().item(), 0.)
                self.assertGreater(field.classifier.bias.grad.abs().sum().item(), 0.)
                if step == 25000:
                    self.assertGreater(field.classifier.weight.grad.abs().sum().item(), 0.)
                    self.assertTrue((point_features.grad.abs().sum(dim=-1) > 0).all())
                else:
                    self.assertIsNone(point_features.grad)


if __name__ == "__main__":
    unittest.main()
