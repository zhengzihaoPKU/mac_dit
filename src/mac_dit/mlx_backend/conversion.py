"""Diffusers DiT 与 MLX checkpoint 之间的转换和加载。"""

import json
from pathlib import Path

import mlx.core as mx
import torch
from diffusers import Transformer2DModel

from .config import MlxDiTConfig, MlxQuantizationConfig
from .model import MlxDiT
from .operators import quantize_linear_modules


FORMAT = "mac_dit_mlx"
FORMAT_VERSION = 1
WEIGHTS_FILENAME = "dit.safetensors"
MANIFEST_FILENAME = "mlx_config.json"


def _to_mlx(tensor, *, transpose_conv=False):
    tensor = tensor.detach().cpu()
    if transpose_conv:
        # PyTorch OIHW -> MLX OHWI。
        tensor = tensor.permute(0, 2, 3, 1)
    return mx.array(tensor.contiguous().numpy())


def diffusers_weight_mapping(transformer):
    """返回 `(MLX 参数名, mx.array)`，所有重命名集中在此函数中。"""
    source = transformer.state_dict()
    weights = [
        (
            "pos_embed.proj.weight",
            _to_mlx(source["pos_embed.proj.weight"], transpose_conv=True),
        ),
        ("pos_embed.proj.bias", _to_mlx(source["pos_embed.proj.bias"])),
        ("pos_embed.position", _to_mlx(transformer.pos_embed.pos_embed)),
    ]

    direct_suffixes = (
        "norm1.emb.timestep_embedder.linear_1.weight",
        "norm1.emb.timestep_embedder.linear_1.bias",
        "norm1.emb.timestep_embedder.linear_2.weight",
        "norm1.emb.timestep_embedder.linear_2.bias",
        "norm1.emb.class_embedder.embedding_table.weight",
        "norm1.linear.weight",
        "norm1.linear.bias",
        "attn1.to_q.weight",
        "attn1.to_q.bias",
        "attn1.to_k.weight",
        "attn1.to_k.bias",
        "attn1.to_v.weight",
        "attn1.to_v.bias",
    )

    for index in range(len(transformer.transformer_blocks)):
        source_prefix = f"transformer_blocks.{index}."
        for suffix in direct_suffixes:
            name = source_prefix + suffix
            weights.append((name, _to_mlx(source[name])))

        renamed = {
            "attn1.to_out.weight": "attn1.to_out.0.weight",
            "attn1.to_out.bias": "attn1.to_out.0.bias",
            "ff.proj.weight": "ff.net.0.proj.weight",
            "ff.proj.bias": "ff.net.0.proj.bias",
            "ff.out.weight": "ff.net.2.weight",
            "ff.out.bias": "ff.net.2.bias",
        }
        for target_suffix, source_suffix in renamed.items():
            weights.append(
                (
                    source_prefix + target_suffix,
                    _to_mlx(source[source_prefix + source_suffix]),
                )
            )

    for name in (
        "proj_out_1.weight",
        "proj_out_1.bias",
        "proj_out_2.weight",
        "proj_out_2.bias",
    ):
        weights.append((name, _to_mlx(source[name])))
    return weights


def convert_transformer(transformer, output_dir, *, quantization=None, model_id=None):
    """将内存中的 Diffusers Transformer 转换并保存为 MLX checkpoint。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = MlxDiTConfig.from_diffusers_config(transformer.config)
    model = MlxDiT(config)
    model.load_weights(diffusers_weight_mapping(transformer), strict=True)
    model.eval()
    mx.eval(model.parameters())

    quantized_paths = ()
    if quantization is not None:
        quantized_paths = quantize_linear_modules(model, quantization)
        mx.eval(model.parameters())

    model.save_weights(str(output_dir / WEIGHTS_FILENAME))
    manifest = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "source_model": model_id,
        "model": config.to_dict(),
        "quantization": quantization.to_dict() if quantization else None,
        "quantized_layers": list(quantized_paths),
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model, manifest


def convert_pretrained(
    model_id,
    cache_dir,
    output_dir,
    *,
    quantization=None,
):
    """只加载 Diffusers Transformer，然后转换为 MLX；不会加载 VAE。"""
    transformer = Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        cache_dir=cache_dir,
        dtype=torch.float16,
    ).eval()
    return convert_transformer(
        transformer,
        output_dir,
        quantization=quantization,
        model_id=model_id,
    )


def load_mlx_transformer(directory):
    """根据 manifest 重建网络结构，再加载 FP16 或量化 MLX 权重。"""
    directory = Path(directory)
    manifest = json.loads(
        (directory / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if manifest.get("format") != FORMAT or manifest.get("version") != FORMAT_VERSION:
        raise ValueError("不是受支持的 mac_dit MLX checkpoint")

    config = MlxDiTConfig(**manifest["model"])
    model = MlxDiT(config)
    model.eval()
    quantization_data = manifest.get("quantization")
    if quantization_data:
        quantization = MlxQuantizationConfig.from_dict(quantization_data)
        quantized_paths = quantize_linear_modules(model, quantization)
        expected_paths = tuple(manifest.get("quantized_layers", ()))
        if expected_paths and quantized_paths != expected_paths:
            raise ValueError("MLX checkpoint 的量化层列表与当前模型不一致")

    model.load_weights(str(directory / WEIGHTS_FILENAME), strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, manifest
