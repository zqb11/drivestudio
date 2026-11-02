"""CPU checks for identity propagation, independent of the CUDA renderer."""

import unittest
from importlib import import_module

import torch
import torch.nn.functional as F

spatial_id_kl = import_module("models.3d_id_loss").spatial_id_kl


class SpatialIdLossTest(unittest.TestCase):
    def test_matches_dial_kl_and_backpropagates_to_both_predictions(self):
        means = torch.tensor([[0., 0., 0.], [0.1, 0., 0.]])
        logits = torch.tensor([[0., 4.], [2., 0.]], requires_grad=True)
        loss, stats = spatial_id_kl(
            means, logits, torch.tensor([1, 0]), torch.ones(2), torch.tensor([1, 1]),
        )
        log_probs = F.log_softmax(logits, dim=-1)
        expected = (log_probs[0].exp() * (log_probs[0] - log_probs[1])).sum() / 2
        torch.testing.assert_close(loss, expected)
        self.assertEqual(stats, {"num_anchors": 1, "num_candidates": 1, "num_pairs": 1})
        loss.backward()
        self.assertTrue((logits.grad.abs().sum(dim=-1) > 0).all())

    def test_nearest_radius_and_instance_label_gates(self):
        means = torch.tensor([
            [0., 0., 0.], [0.1, 0., 0.], [0.2, 0., 0.],
            [2., 0., 0.], [0.05, 0., 0.], [0.01, 0., 0.],
        ])
        logits = torch.tensor([
            [0., 4., 0.], [2., 0., 0.], [1., 1., 0.],
            [2., 0., 0.], [2., 0., 0.], [2., 0., 0.],
        ], requires_grad=True)
        loss, stats = spatial_id_kl(
            means, logits, torch.tensor([1, 0, 0, 0, 0, 0]),
            torch.tensor([1., 1., 1., 1., 1., 0.]),
            torch.tensor([1, 1, 1, 1, 2, 1]),
            {"k": 1, "anchor_chunk_size": 1, "candidate_chunk_size": 1},
        )
        self.assertEqual(stats["num_pairs"], 1)
        loss.backward()
        self.assertTrue(logits.grad[1].abs().sum() > 0)
        self.assertEqual(logits.grad[2:].abs().sum().item(), 0.)

    def test_empty_or_unreliable_selections_allow_backward(self):
        for structural, projected, coordinates, raw_logits in [
            ([0, 0], [1, 1], [[0., 0., 0.], [.1, 0., 0.]], [[0., 4.], [2., 0.]]),
            ([1, 1], [1, 1], [[0., 0., 0.], [.1, 0., 0.]], [[0., 4.], [0., 4.]]),
            ([1, 0], [-1, 1], [[0., 0., 0.], [.1, 0., 0.]], [[0., 4.], [2., 0.]]),
            ([1, 0], [1, 1], [[0., 0., 0.], [3., 0., 0.]], [[0., 4.], [2., 0.]]),
            ([1, 0], [1, 1], [[0., 0., 0.], [.1, 0., 0.]], [[0., 0.], [2., 0.]]),
        ]:
            with self.subTest(structural=structural, projected=projected, logits=raw_logits):
                logits = torch.tensor(raw_logits, requires_grad=True)
                loss, stats = spatial_id_kl(
                    torch.tensor(coordinates), logits, torch.tensor(structural),
                    torch.ones(2), torch.tensor(projected),
                )
                self.assertEqual(loss.item(), 0.)
                self.assertEqual(stats["num_pairs"], 0)
                loss.backward()
                self.assertEqual(logits.grad.abs().sum().item(), 0.)

    def test_empty_input_and_small_candidate_set(self):
        logits = torch.empty((0, 2), requires_grad=True)
        loss, stats = spatial_id_kl(
            torch.empty((0, 3)), logits, torch.empty(0, dtype=torch.long),
            torch.empty(0), torch.empty(0, dtype=torch.long),
        )
        loss.backward()
        self.assertEqual(stats["num_pairs"], 0)

    def test_sampling_caps_limit_work(self):
        logits = torch.tensor([[0., 4.]] * 6 + [[2., 0.]] * 8, requires_grad=True)
        loss, stats = spatial_id_kl(
            torch.zeros((14, 3)), logits,
            torch.tensor([1] * 6 + [0] * 8), torch.ones(14), torch.ones(14, dtype=torch.long),
            {"max_anchors": 2, "max_candidates": 3, "k": 10,
             "anchor_chunk_size": 1, "candidate_chunk_size": 2},
        )
        self.assertEqual(stats, {"num_anchors": 2, "num_candidates": 3, "num_pairs": 6})
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
