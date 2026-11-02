"""CPU checks for bounded-memory evaluation and metric aggregation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from models import video_utils


class MemoryWriter:
    def __init__(self):
        self.frames = []
        self.close_count = 0

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.close_count += 1


class MockSplit:
    split = "full"

    def __len__(self):
        return 12


class StreamingEvaluationTest(unittest.TestCase):
    def make_results(self, frame):
        metric_counts = {
            "psnr": 6, "ssim": 6, "lpips": 6,
            "occupied_psnr": 6, "occupied_ssim": 6,
            "masked_psnr": 1 if frame == 0 else 3,
            "masked_ssim": 1 if frame == 0 else 3,
            "human_psnr": 0, "human_ssim": 0,
            "vehicle_psnr": 0, "vehicle_ssim": 0,
        }
        return {
            "psnr": 10. if frame == 0 else 20.,
            "ssim": .2 if frame == 0 else .8,
            "lpips": .1 if frame == 0 else .3,
            "occupied_psnr": 11. if frame == 0 else 21.,
            "occupied_ssim": .3 if frame == 0 else .7,
            "masked_psnr": 30. if frame == 0 else 50.,
            "masked_ssim": .3 if frame == 0 else .9,
            "human_psnr": -1., "human_ssim": -1.,
            "vehicle_psnr": -1., "vehicle_ssim": -1.,
            "_metric_counts": metric_counts,
            "cam_names": [f"cam{camera}" for camera in range(6)],
            "cam_ids": list(range(6)),
            "Background_rgbs": [np.full((8, 8, 3), .25 if frame == 0 else .75, np.float32) for _ in range(6)],
            "Dynamic_rgbs": [np.full((8, 8, 3), .5 if frame == 0 else .1, np.float32) for _ in range(6)],
        }

    def layout(self, frames, camera_names):
        self.assertEqual(camera_names, [f"cam{camera}" for camera in range(6)])
        self.assertEqual(len(frames), 6)
        for frame in frames:
            self.assertTrue(np.issubdtype(frame.dtype, np.floating))
            self.assertGreaterEqual(frame.min(), 0.)
            self.assertLessEqual(frame.max(), 1.)
        return np.concatenate(frames, axis=1)

    def execute(self, separate=True, render_effect=None):
        trainer = SimpleNamespace(set_eval=lambda: None)
        opened = {}

        def make_writer(path, **kwargs):
            self.assertNotIn(path, opened)
            opened[path] = MemoryWriter()
            return opened[path]

        effect = render_effect or [self.make_results(0), self.make_results(1)]
        with patch.object(video_utils, "render", side_effect=effect) as render_mock:
            with patch.object(video_utils.imageio, "get_writer", side_effect=make_writer):
                metrics, midpoint = video_utils.render_images_streaming(
                    trainer=trainer, dataset=MockSplit(),
                    save_pth="/tmp/test_streaming_eval.mp4", layout=self.layout,
                    num_timestamps=2, num_cams=6,
                    keys=["Background_rgbs", "Dynamic_rgbs", "Missing_rgbs"],
                    save_seperate_video=separate, fps=10,
                )
        return metrics, midpoint, opened, render_mock

    def test_metric_counts_weighted_means_and_empty_masks(self):
        metrics, _, _, _ = self.execute()
        self.assertAlmostEqual(metrics["psnr"], 15.)
        self.assertAlmostEqual(metrics["ssim"], .5)
        self.assertAlmostEqual(metrics["lpips"], .2)
        self.assertAlmostEqual(metrics["occupied_psnr"], 16.)
        self.assertAlmostEqual(metrics["masked_psnr"], 45.)
        self.assertAlmostEqual(metrics["masked_ssim"], .75)
        self.assertEqual(metrics["human_psnr"], -1.)
        self.assertEqual(metrics["human_ssim"], -1.)
        self.assertEqual(metrics["vehicle_psnr"], -1.)
        self.assertEqual(metrics["vehicle_ssim"], -1.)

    def test_timestep_vis_indices_and_float_to_uint8_encoding(self):
        _, midpoint, opened, render_mock = self.execute()
        self.assertEqual([call.kwargs["vis_indices"] for call in render_mock.call_args_list],
                         [list(range(6)), list(range(6, 12))])
        background = opened["/tmp/test_streaming_eval_Background_rgbs.mp4"]
        self.assertEqual(len(background.frames), 2)
        self.assertEqual(background.frames[0].dtype, np.uint8)
        self.assertTrue((background.frames[0] == 63).all())
        self.assertTrue((background.frames[1] == 191).all())
        np.testing.assert_array_equal(midpoint["Background_rgbs"], background.frames[1])

    def test_writers_open_only_existing_keys_and_close_once(self):
        _, midpoint, opened, _ = self.execute()
        self.assertEqual(set(opened), {
            "/tmp/test_streaming_eval_Background_rgbs.mp4",
            "/tmp/test_streaming_eval_Dynamic_rgbs.mp4",
        })
        self.assertEqual(set(midpoint), {"Background_rgbs", "Dynamic_rgbs"})
        self.assertTrue(all(writer.close_count == 1 for writer in opened.values()))

    def test_concatenated_writer_and_midpoint(self):
        _, midpoint, opened, _ = self.execute(separate=False)
        self.assertEqual(set(opened), {"/tmp/test_streaming_eval.mp4"})
        writer = opened["/tmp/test_streaming_eval.mp4"]
        self.assertEqual(len(writer.frames), 2)
        self.assertEqual(writer.frames[0].shape, (16, 48, 3))
        self.assertEqual(writer.close_count, 1)
        np.testing.assert_array_equal(midpoint["concatenated_frame"], writer.frames[1])

    def test_writer_closes_when_rendering_fails(self):
        writer = MemoryWriter()
        trainer = SimpleNamespace(set_eval=lambda: None)
        with patch.object(video_utils, "render", side_effect=[self.make_results(0), RuntimeError("render failed")]):
            with patch.object(video_utils.imageio, "get_writer", return_value=writer):
                with self.assertRaisesRegex(RuntimeError, "render failed"):
                    video_utils.render_images_streaming(
                        trainer=trainer, dataset=MockSplit(),
                        save_pth="/tmp/test_streaming_failure.mp4", layout=self.layout,
                        num_timestamps=2, num_cams=6, keys=["Background_rgbs"],
                    )
        self.assertEqual(writer.close_count, 1)

    def test_actual_render_reports_per_metric_valid_image_counts(self):
        generator = torch.Generator().manual_seed(17)
        target = torch.rand((8, 8, 3), generator=generator)

        class Dataset:
            split = "full"

            def __len__(self):
                return 2

            def get_image(self, index, downscale):
                images = {
                    "pixels": target,
                    "sky_masks": torch.ones(8, 8) if index == 0 else torch.zeros(8, 8),
                    "dynamic_masks": torch.ones(8, 8) if index == 0 else torch.zeros(8, 8),
                    "human_masks": torch.zeros(8, 8),
                    "vehicle_masks": torch.ones(8, 8),
                }
                cameras = {"cam_name": "CAM_FRONT", "cam_id": torch.tensor([0])}
                return images, cameras

        class Trainer:
            def _get_downscale_factor(self):
                return 1

            def __call__(self, image_infos, cam_infos):
                return {"rgb": image_infos["pixels"] * .9,
                        "depth": torch.ones((8, 8, 1)), "opacity": torch.ones((8, 8, 1))}

            def lpips(self, prediction, reference):
                return (prediction - reference).square().mean()

        with patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self):
            results = video_utils.render(Dataset(), trainer=Trainer(), compute_metrics=True)
        self.assertEqual(results["_metric_counts"], {
            "psnr": 2, "ssim": 2, "lpips": 2,
            "occupied_psnr": 1, "occupied_ssim": 1,
            "masked_psnr": 1, "masked_ssim": 1,
            "human_psnr": 0, "human_ssim": 0,
            "vehicle_psnr": 2, "vehicle_ssim": 2,
        })
        self.assertEqual(results["human_psnr"], -1)

    def test_ssim_full_scalar_and_mask_map_match_original_calls(self):
        rng = np.random.default_rng(47)
        reference = rng.random((15, 17, 3), dtype=np.float32)
        prediction = np.clip(reference * .8 + .1, 0, 1)
        full_scalar, shared_map = ssim(prediction, reference, data_range=1., channel_axis=-1, full=True)
        scalar = ssim(prediction, reference, data_range=1., channel_axis=-1)
        self.assertEqual(full_scalar, scalar)
        masks = [np.indices((15, 17))[0] < 5, np.indices((15, 17))[1] > 9]
        for mask in masks:
            old_map = ssim(prediction, reference, data_range=1., channel_axis=-1, full=True)[1]
            self.assertEqual(shared_map[mask].mean(), old_map[mask].mean())


if __name__ == "__main__":
    unittest.main()
