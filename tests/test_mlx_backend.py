import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from diffusers import Transformer2DModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_dit.mlx_backend.config import MlxDiTConfig, MlxQuantizationConfig


class MlxConfigTests(unittest.TestCase):
    def test_default_dit_dimensions(self):
        config = MlxDiTConfig()
        self.assertEqual(config.hidden_size, 1152)
        self.assertEqual(config.patch_tokens, 256)

    def test_quantization_scope(self):
        config = MlxQuantizationConfig(bits=4, group_size=128)
        self.assertTrue(config.matches("transformer_blocks.0.attn1.to_q"))
        self.assertTrue(config.matches("transformer_blocks.0.ff.proj"))
        self.assertFalse(config.matches("transformer_blocks.0.norm1.linear"))


class MlxRuntimeTests(unittest.TestCase):
    def setUp(self):
        if not torch.backends.mps.is_available():
            self.skipTest("当前测试进程不可访问 Metal GPU")
        import mlx.core as mx

        self.mx = mx

    @staticmethod
    def make_transformer(hidden_size=8):
        return Transformer2DModel(
            num_attention_heads=2 if hidden_size == 8 else 4,
            attention_head_dim=4 if hidden_size == 8 else 8,
            in_channels=4,
            out_channels=8,
            num_layers=1,
            attention_bias=True,
            sample_size=4,
            patch_size=2,
            activation_fn="gelu-approximate",
            num_embeds_ada_norm=10,
            norm_type="ada_norm_zero",
            norm_elementwise_affine=False,
        ).eval()

    def test_diffusers_and_mlx_forward_match(self):
        from mac_dit.mlx_backend.conversion import convert_transformer

        torch.manual_seed(17)
        transformer = self.make_transformer()
        sample = torch.randn(2, 4, 4, 4)
        timesteps = torch.tensor([5, 7])
        labels = torch.tensor([2, 10])
        expected = transformer(
            sample,
            timestep=timesteps,
            class_labels=labels,
        ).sample.detach().numpy()

        with tempfile.TemporaryDirectory() as directory:
            model, _ = convert_transformer(transformer, directory)
            actual = model(
                self.mx.array(sample.numpy()),
                self.mx.array(timesteps.numpy()),
                self.mx.array(labels.numpy()),
            )
            self.mx.eval(actual)

        np.testing.assert_allclose(np.asarray(actual), expected, atol=2e-5, rtol=2e-5)

    def test_quantized_checkpoint_uses_six_qmm_layers(self):
        from mac_dit.mlx_backend.conversion import (
            convert_transformer,
            load_mlx_transformer,
        )

        torch.manual_seed(19)
        transformer = self.make_transformer(hidden_size=32)
        sample = torch.randn(1, 4, 4, 4)
        timesteps = torch.tensor([3])
        labels = torch.tensor([1])
        quantization = MlxQuantizationConfig(bits=4, group_size=32)

        with tempfile.TemporaryDirectory() as directory:
            model, manifest = convert_transformer(
                transformer,
                directory,
                quantization=quantization,
            )
            expected = model(
                self.mx.array(sample.numpy()),
                self.mx.array(timesteps.numpy()),
                self.mx.array(labels.numpy()),
            )
            loaded, _ = load_mlx_transformer(directory)
            actual = loaded(
                self.mx.array(sample.numpy()),
                self.mx.array(timesteps.numpy()),
                self.mx.array(labels.numpy()),
            )
            self.mx.eval(expected, actual)

            self.assertEqual(len(manifest["quantized_layers"]), 6)
            self.assertTrue((Path(directory) / "dit.safetensors").is_file())

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
