import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from diffusers import Transformer2DModel
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mac_dit.cli import build_parser, generation_config_from_args
from mac_dit.config import GenerationConfig
from mac_dit.pipeline import generate_image
from mac_dit.quantization.algorithms import dequantize_weight, quantize_weight
from mac_dit.quantization.backends import available_backends, get_backend
from mac_dit.quantization.config import QuantizationConfig
from mac_dit.quantization.modules import QuantizedLinear
from mac_dit.quantization.serialization import (
    load_quantized_model,
    save_quantized_model,
)
from mac_dit.quantization.transform import quantize_model


class Attention(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.to_q = nn.Linear(features, features)

    def forward(self, inputs):
        return self.to_q(inputs)


class Block(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.norm = nn.Linear(features, features)
        self.attn1 = Attention(features)
        self.ff = nn.Sequential(nn.Linear(features, features))

    def forward(self, inputs):
        return self.norm(inputs) + self.attn1(inputs) + self.ff(inputs)


class ToyTransformer(nn.Module):
    def __init__(self, features=16):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([Block(features)])

    def forward(self, inputs):
        return self.transformer_blocks[0](inputs)


class FakePipeline:
    def __call__(
        self,
        class_labels,
        generator,
        num_inference_steps,
    ):
        pixel = int(torch.randint(0, 256, (1,), generator=generator).item())
        image = Image.new("L", (2, 2), color=pixel)
        return SimpleNamespace(images=[image])


class WeightQuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.weight = torch.randn(7, 13)

    def test_int8_round_trip(self):
        config = QuantizationConfig(mode="int8", include_patterns=())
        quantized = quantize_weight(self.weight, config)
        restored = dequantize_weight(quantized, dtype=torch.float32)

        self.assertEqual(quantized.qweight.dtype, torch.int8)
        self.assertEqual(restored.shape, self.weight.shape)
        self.assertLess((restored - self.weight).abs().max().item(), 0.02)

    def test_int4_pack_and_round_trip(self):
        config = QuantizationConfig(
            mode="int4",
            group_size=8,
            include_patterns=(),
        )
        quantized = quantize_weight(self.weight, config)
        restored = dequantize_weight(quantized, dtype=torch.float32)

        self.assertEqual(quantized.qweight.dtype, torch.uint8)
        self.assertEqual(quantized.padded_in_features, 16)
        self.assertEqual(quantized.qweight.shape, (7, 8))
        self.assertLess((restored - self.weight).abs().max().item(), 0.3)

    def test_invalid_int4_group_size(self):
        with self.assertRaises(ValueError):
            QuantizationConfig(mode="int4", group_size=7)

    def test_backend_registry_exposes_reference_and_metal(self):
        self.assertIn("reference", available_backends())
        self.assertIn("metal", available_backends())

        quantized = quantize_weight(
            self.weight,
            QuantizationConfig(mode="int8", backend="metal"),
        )
        self.assertFalse(
            get_backend("metal").supports(quantized, torch.device("cpu"))
        )


class QuantizationCliTests(unittest.TestCase):
    def test_quantize_only_uses_model_quantized_directory(self):
        args = build_parser().parse_args(
            ["--quantization", "int4", "--group-size", "64", "--quantize-only"]
        )
        config = generation_config_from_args(args)

        self.assertEqual(
            config.save_quantized_dir,
            PROJECT_ROOT / "model" / "quantized" / "DiT-XL-2-256-int4-g64",
        )

    def test_auto_save_requires_quantization(self):
        args = build_parser().parse_args(["--save-quantized"])
        with self.assertRaises(ValueError):
            generation_config_from_args(args)

    def test_metal_source_contains_direct_quantized_linear_kernels(self):
        kernel_path = (
            PROJECT_ROOT
            / "src/mac_dit/quantization/backends/kernels/dequantize.metal"
        )
        source = kernel_path.read_text(encoding="utf-8")
        self.assertIn("kernel void linear_int8", source)
        self.assertIn("kernel void linear_int4", source)


class ModelTransformTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.model = ToyTransformer().eval()
        self.inputs = torch.randn(2, 3, 16)

    def test_default_scope_keeps_norm_in_float(self):
        baseline = self.model(self.inputs)
        report = quantize_model(self.model, QuantizationConfig(mode="int8"))
        result = self.model(self.inputs)
        block = self.model.transformer_blocks[0]

        self.assertIsInstance(block.norm, nn.Linear)
        self.assertIsInstance(block.attn1.to_q, QuantizedLinear)
        self.assertIsInstance(block.ff[0], QuantizedLinear)
        self.assertEqual(report.total_linear_layers, 3)
        self.assertEqual(report.quantized_layers, 2)
        self.assertGreater(report.compression_ratio, 1.5)
        torch.testing.assert_close(result, baseline, atol=0.04, rtol=0.04)

    def test_quantized_checkpoint_round_trip(self):
        baseline_model = copy.deepcopy(self.model)
        override_model = copy.deepcopy(self.model)
        quantize_model(
            self.model,
            QuantizationConfig(mode="int4", group_size=8),
        )
        expected = self.model(self.inputs)

        with tempfile.TemporaryDirectory() as directory:
            save_quantized_model(self.model, directory, model_id="test/dit")
            report = load_quantized_model(baseline_model, directory)
            actual = baseline_model(self.inputs)
            load_quantized_model(override_model, directory, backend="metal")

        self.assertEqual(report.quantized_layers, 2)
        self.assertEqual(
            override_model.transformer_blocks[0].attn1.to_q.quantization_config.backend,
            "metal",
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_diffusers_transformer_forward_after_quantization(self):
        model = Transformer2DModel(
            num_attention_heads=2,
            attention_head_dim=4,
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
        hidden_states = torch.randn(1, 4, 4, 4)
        timestep = torch.tensor([5])
        class_labels = torch.tensor([2])
        expected = model(
            hidden_states,
            timestep=timestep,
            class_labels=class_labels,
        ).sample

        report = quantize_model(
            model,
            QuantizationConfig(mode="int8", group_size=4),
        )
        actual = model(
            hidden_states,
            timestep=timestep,
            class_labels=class_labels,
        ).sample

        self.assertEqual(report.quantized_layers, 6)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)


class ReproducibilityTests(unittest.TestCase):
    def test_same_seed_produces_same_image(self):
        with tempfile.TemporaryDirectory() as directory:
            config = GenerationConfig(
                device="cpu",
                output_dir=directory,
                seed=123,
                inference_steps=1,
            )
            first = generate_image(FakePipeline(), config)
            first_pixel = Image.open(first.image_path).getpixel((0, 0))
            second = generate_image(FakePipeline(), config)
            second_pixel = Image.open(second.image_path).getpixel((0, 0))

        self.assertEqual(first_pixel, second_pixel)
        self.assertIn("seed123", first.image_path.name)


if __name__ == "__main__":
    unittest.main()
