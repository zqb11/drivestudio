"""CPU checks for per-Gaussian identity parameters across real model operations."""

import copy
import unittest

import torch
from omegaconf import OmegaConf

from models.gaussians.basics import dup_in_optim, remove_from_optim
from models.gaussians.vanilla import VanillaGaussians
from models.nodes.rigid import RigidNodes


class IdentityLifecycleTest(unittest.TestCase):
    def make_model(self, model_type=VanillaGaussians, embedding_dim=8):
        ctrl = OmegaConf.create({
            "sh_degree": 0,
            "id_embedding_dim": embedding_dim,
            "warmup_steps": 0,
            "reset_alpha_interval": 1000,
            "refine_interval": 10,
            "stop_split_at": 150,
            "densify_grad_thresh": .1,
            "densify_size_thresh": .1,
            "stop_screen_size_at": 0,
            "n_split_samples": 2,
            "cull_alpha_thresh": .05,
            "cull_scale_thresh": 100,
            "cull_screen_size": 1,
            "cull_out_of_bound": False,
        })
        model = model_type(
            class_name="Test", ctrl=ctrl, num_train_images=1,
            scene_scale=1., device=torch.device("cpu"),
        )
        points = torch.tensor([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
            [2., 0., 0.], [3., 0., 0.], [2., 1., 0.], [2., 0., 1.],
        ])
        colors = torch.full_like(points, .5)
        if model_type is VanillaGaussians:
            model.create_from_pcd(points, colors)
        else:
            instances = {}
            for instance in range(2):
                selection = slice(instance * 4, (instance + 1) * 4)
                instances[instance] = {
                    "pts": points[selection], "colors": colors[selection],
                    "poses": torch.eye(4).repeat(2, 1, 1),
                    "size": torch.full((3,), 10.),
                    "frame_info": torch.ones(2, dtype=torch.bool), "num_pts": 4,
                }
            model.create_from_pcd(instances)
        return model

    def make_optimizer(self, model):
        return torch.optim.Adam([
            {"params": params, "name": name}
            for name, params in model.get_gaussian_param_groups().items()
        ])

    def assert_optimizer_aligned(self, model, optimizer):
        groups = model.get_gaussian_param_groups()
        for group in optimizer.param_groups:
            expected = groups[group["name"]][0]
            self.assertIs(group["params"][0], expected)
            state = optimizer.state[expected]
            self.assertEqual(state["exp_avg"].shape, expected.shape)
            self.assertEqual(state["exp_avg_sq"].shape, expected.shape)

    def test_refinement_preserves_parent_identity_and_optimizer_state(self):
        for model_type in (VanillaGaussians, RigidNodes):
            with self.subTest(model=model_type.__name__):
                model = self.make_model(model_type)
                model._id_features.data.copy_(torch.arange(64.).reshape(8, 8))
                model._scales.data[:4].fill_(torch.log(torch.tensor(.2)))
                model._scales.data[4:].fill_(torch.log(torch.tensor(.01)))
                optimizer = self.make_optimizer(model)
                sum(parameter.sum() for group in optimizer.param_groups
                    for parameter in group["params"]).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                parent_features = model._id_features.detach().clone()
                parent_momentum = optimizer.state[model._id_features]["exp_avg"].clone()
                model.step = 100
                model.xys_grad_norm = torch.ones(8)
                model.vis_counts = torch.ones(8)
                model.max_2Dsize = torch.zeros(8)
                model.refinement_after(100, optimizer)

                self.assertEqual(model.num_points, 20)
                for parent, features in enumerate(parent_features):
                    descendants = (model._id_features.detach() == features).all(dim=1)
                    self.assertEqual(descendants.sum().item(), 3 if parent < 4 else 2)
                    if model_type is RigidNodes:
                        self.assertTrue((model.point_ids[descendants, 0] == parent // 4).all())
                self.assert_optimizer_aligned(model, optimizer)
                momentum = optimizer.state[model._id_features]["exp_avg"]
                torch.testing.assert_close(momentum[:8], parent_momentum)
                self.assertEqual(torch.count_nonzero(momentum[8:]).item(), 0)

                before_cull = model._id_features.detach().clone()
                before_momentum = momentum.clone()
                point_ids = model.point_ids.clone() if model_type is RigidNodes else None
                retained = torch.ones(model.num_points, dtype=torch.bool)
                retained[[0, 10]] = False
                model._opacities.data[~retained] = -100.
                model.step = 200
                model.refinement_after(200, optimizer)
                self.assertEqual(model.num_points, 18)
                torch.testing.assert_close(model._id_features, before_cull[retained])
                torch.testing.assert_close(
                    optimizer.state[model._id_features]["exp_avg"], before_momentum[retained],
                )
                if point_ids is not None:
                    self.assertTrue(torch.equal(model.point_ids, point_ids[retained]))
                self.assert_optimizer_aligned(model, optimizer)
                model._id_features.sum().backward()
                optimizer.step()

    def test_new_and_legacy_checkpoints_keep_strict_loading(self):
        for model_type in (VanillaGaussians, RigidNodes):
            with self.subTest(model=model_type.__name__):
                original = self.make_model(model_type)
                original._id_features.data.copy_(torch.arange(64.).reshape(8, 8))
                state = copy.deepcopy(original.state_dict())
                restored = self.make_model(model_type)
                result = restored.load_state_dict(copy.deepcopy(state), strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
                torch.testing.assert_close(restored._id_features, original._id_features)

                legacy = copy.deepcopy(state)
                del legacy["_id_features"]
                result = restored.load_state_dict(copy.deepcopy(legacy), strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(restored._id_features.shape, (8, 8))
                self.assertTrue(torch.isfinite(restored._id_features).all())
                torch.testing.assert_close(restored._means, original._means)
                broken = copy.deepcopy(legacy)
                del broken["_opacities"]
                with self.assertRaises(RuntimeError):
                    restored.load_state_dict(broken, strict=True)
                unexpected = copy.deepcopy(legacy)
                unexpected["unexpected_parameter"] = torch.zeros(1)
                with self.assertRaises(RuntimeError):
                    restored.load_state_dict(unexpected, strict=True)

    def test_disabled_identity_keeps_legacy_parameter_groups(self):
        for model_type in (VanillaGaussians, RigidNodes):
            with self.subTest(model=model_type.__name__):
                model = self.make_model(model_type, embedding_dim=0)
                self.assertFalse(hasattr(model, "_id_features"))
                self.assertNotIn("Test#id_features", model.get_gaussian_param_groups())
                self.assertNotIn("_id_features", model.state_dict())
                model.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)

    def test_optimizer_parameter_replacement_before_first_gradient(self):
        original = torch.nn.Parameter(torch.arange(32.).reshape(4, 8))
        optimizer = torch.optim.Adam([{"params": [original], "name": "id_features"}])
        expanded = torch.nn.Parameter(torch.cat([original.detach(), original[:1].detach()]))
        dup_in_optim(optimizer, torch.tensor([0]), {"id_features": [expanded]}, n=1)
        self.assertIs(optimizer.param_groups[0]["params"][0], expanded)
        self.assertFalse(optimizer.state)
        deleted = torch.tensor([False, True, False, False, False])
        retained = torch.nn.Parameter(expanded.detach()[~deleted])
        remove_from_optim(optimizer, deleted, {"id_features": [retained]})
        self.assertIs(optimizer.param_groups[0]["params"][0], retained)
        self.assertFalse(optimizer.state)
        before_step = retained.detach().clone()
        retained.sum().backward()
        optimizer.step()
        self.assertTrue((retained.detach() < before_step).all())
        self.assertEqual(optimizer.state[retained]["exp_avg"].shape, retained.shape)


if __name__ == "__main__":
    unittest.main()
