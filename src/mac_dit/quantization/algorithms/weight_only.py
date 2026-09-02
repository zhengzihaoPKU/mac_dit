import torch
import torch.nn.functional as F

from ..config import QuantizationConfig
from ..types import QuantizedWeight


INT8_MAX = 127
INT4_MAX = 7
INT4_OFFSET = 8


def _safe_scale(max_abs, quant_max):
    scale = max_abs / quant_max
    return torch.clamp(scale, min=torch.finfo(torch.float32).eps)


def _quantize_int8(weight, config):
    source = weight.detach().to(dtype=torch.float32)
    scales = _safe_scale(source.abs().amax(dim=1, keepdim=True), INT8_MAX)
    qweight = torch.round(source / scales).clamp(-INT8_MAX, INT8_MAX)
    return QuantizedWeight(
        qweight=qweight.to(torch.int8),
        scales=scales.to(torch.float16),
        original_shape=tuple(weight.shape),
        padded_in_features=weight.shape[1],
        config=config,
    )


def _pack_int4(values):
    unsigned = (values + INT4_OFFSET).to(torch.uint8)
    low = unsigned[..., 0::2]
    high = unsigned[..., 1::2] << 4
    return low | high


def _unpack_int4(packed):
    low = (packed & 0x0F).to(torch.int16) - INT4_OFFSET
    high = (packed >> 4).to(torch.int16) - INT4_OFFSET
    return torch.stack((low, high), dim=-1).flatten(-2)


def _quantize_int4(weight, config):
    source = weight.detach().to(dtype=torch.float32)
    out_features, in_features = source.shape
    padding = (-in_features) % config.group_size
    if padding:
        source = F.pad(source, (0, padding))

    padded_in_features = source.shape[1]
    grouped = source.reshape(out_features, -1, config.group_size)
    scales = _safe_scale(grouped.abs().amax(dim=-1, keepdim=True), INT4_MAX)
    quantized = torch.round(grouped / scales).clamp(-INT4_MAX, INT4_MAX)
    packed = _pack_int4(quantized.to(torch.int8).reshape(out_features, -1))
    return QuantizedWeight(
        qweight=packed,
        scales=scales.squeeze(-1).to(torch.float16),
        original_shape=tuple(weight.shape),
        padded_in_features=padded_in_features,
        config=config,
    )


def quantize_weight(weight, config):
    """将二维 Linear 权重量化为后端无关格式。"""
    if weight.ndim != 2:
        raise ValueError("仅支持二维 Linear 权重")
    if config.mode == "int8":
        return _quantize_int8(weight, config)
    return _quantize_int4(weight, config)


def dequantize_weight(quantized_weight, *, dtype=None):
    """将量化权重恢复到浮点格式，供参考后端和数值测试使用。"""
    output_dtype = dtype or torch.float16
    out_features, in_features = quantized_weight.original_shape

    if quantized_weight.config.mode == "int8":
        weight = quantized_weight.qweight.to(output_dtype)
        weight = weight * quantized_weight.scales.to(output_dtype)
        return weight.reshape(out_features, in_features)

    unpacked = _unpack_int4(quantized_weight.qweight)
    grouped = unpacked.reshape(
        out_features,
        -1,
        quantized_weight.config.group_size,
    ).to(output_dtype)
    weight = grouped * quantized_weight.scales.to(output_dtype).unsqueeze(-1)
    return weight.reshape(out_features, -1)[:, :in_features]
