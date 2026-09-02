import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .config import QuantizationConfig
from .modules import QuantizedLinear
from .transform import QuantizationReport
from .types import tensor_storage_bytes


FORMAT_VERSION = 1
WEIGHTS_FILENAME = "transformer.safetensors"
MANIFEST_FILENAME = "quantization.json"


def _config_to_dict(config):
    return {
        "mode": config.mode,
        "backend": config.backend,
        "group_size": config.group_size,
        "include_patterns": list(config.include_patterns),
        "exclude_patterns": list(config.exclude_patterns),
    }


def _config_from_dict(data, *, backend=None):
    return QuantizationConfig(
        mode=data["mode"],
        backend=backend or data["backend"],
        group_size=data["group_size"],
        include_patterns=tuple(data.get("include_patterns", ())),
        exclude_patterns=tuple(data.get("exclude_patterns", ())),
    )


def save_quantized_model(model, directory, *, model_id=None):
    """保存包含 QuantizedLinear 的完整 Transformer state_dict。"""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            layers.append(
                {
                    "name": name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                    "padded_in_features": module.padded_in_features,
                    "has_bias": module.bias is not None,
                    "config": _config_to_dict(module.quantization_config),
                }
            )

    if not layers:
        raise ValueError("模型中没有可保存的 QuantizedLinear")

    state = {
        name: tensor.detach().to("cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(
        state,
        output_dir / WEIGHTS_FILENAME,
        metadata={"format": "mac_dit_quantized_transformer", "version": str(FORMAT_VERSION)},
    )
    manifest = {
        "format": "mac_dit_quantized_transformer",
        "version": FORMAT_VERSION,
        "model_id": model_id,
        "layers": layers,
    }
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def _replace_submodule(model, name, replacement):
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    parent._modules[child_name] = replacement


def load_quantized_model(model, directory, *, backend=None):
    """将量化 checkpoint 加载到同架构的浮点 Transformer。"""
    input_dir = Path(directory)
    manifest = json.loads(
        (input_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    if manifest.get("version") != FORMAT_VERSION:
        raise ValueError(f"不支持的量化格式版本: {manifest.get('version')}")

    state = load_file(input_dir / WEIGHTS_FILENAME, device="cpu")
    original_bytes = 0
    quantized_bytes = 0
    layer_names = []
    total_linear_layers = sum(
        isinstance(module, torch.nn.Linear) for module in model.modules()
    )

    for layer in manifest["layers"]:
        name = layer["name"]
        original = model.get_submodule(name)
        if not isinstance(original, torch.nn.Linear):
            raise TypeError(f"{name} 不是可替换的 nn.Linear")

        original_bytes += tensor_storage_bytes(original.weight)
        original_bytes += tensor_storage_bytes(original.bias)
        config = _config_from_dict(layer["config"], backend=backend)
        prefix = f"{name}."
        replacement = QuantizedLinear(
            layer["in_features"],
            layer["out_features"],
            config=config,
            qweight=state[f"{prefix}qweight"],
            scales=state[f"{prefix}scales"],
            bias=state.get(f"{prefix}bias"),
            padded_in_features=layer["padded_in_features"],
            zero_points=state.get(f"{prefix}zero_points"),
        )
        _replace_submodule(model, name, replacement)
        quantized_bytes += replacement.storage_bytes
        layer_names.append(name)

    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "量化 checkpoint 与模型不兼容: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    return QuantizationReport(
        total_linear_layers=total_linear_layers,
        quantized_layers=len(layer_names),
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        layer_names=tuple(layer_names),
    )
