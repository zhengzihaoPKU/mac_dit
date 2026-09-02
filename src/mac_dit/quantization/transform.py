from dataclasses import dataclass

from torch import nn

from .config import QuantizationConfig
from .modules import QuantizedLinear
from .types import tensor_storage_bytes


@dataclass(frozen=True, slots=True)
class QuantizationReport:
    total_linear_layers: int
    quantized_layers: int
    original_bytes: int
    quantized_bytes: int
    layer_names: tuple[str, ...]

    @property
    def compression_ratio(self):
        if self.quantized_bytes == 0:
            return 1.0
        return self.original_bytes / self.quantized_bytes

    @property
    def saved_bytes(self):
        return self.original_bytes - self.quantized_bytes


def _matches_layer(name, config):
    included = not config.include_patterns or any(
        pattern in name for pattern in config.include_patterns
    )
    excluded = any(pattern in name for pattern in config.exclude_patterns)
    return included and not excluded


def _replace_submodule(model, name, replacement):
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    parent._modules[child_name] = replacement


def quantize_model(model, config=None):
    """原地替换匹配的 nn.Linear，并返回量化报告。"""
    config = config or QuantizationConfig()
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not isinstance(module, QuantizedLinear)
    ]
    selected = [
        (name, module)
        for name, module in candidates
        if _matches_layer(name, config)
    ]

    original_bytes = 0
    quantized_bytes = 0
    layer_names = []
    for name, module in selected:
        original_bytes += tensor_storage_bytes(module.weight)
        original_bytes += tensor_storage_bytes(module.bias)
        replacement = QuantizedLinear.from_float(module, config)
        _replace_submodule(model, name, replacement)
        quantized_bytes += replacement.storage_bytes
        layer_names.append(name)

    return QuantizationReport(
        total_linear_layers=len(candidates),
        quantized_layers=len(selected),
        original_bytes=original_bytes,
        quantized_bytes=quantized_bytes,
        layer_names=tuple(layer_names),
    )
