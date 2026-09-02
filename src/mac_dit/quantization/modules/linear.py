import torch
from torch import nn

from ..algorithms import quantize_weight
from ..backends import get_backend
from ..config import QuantizationConfig
from ..types import QuantizedWeight, tensor_storage_bytes


class QuantizedLinear(nn.Module):
    """保存低比特权重并把计算委托给量化后端的 Linear。"""

    def __init__(
        self,
        in_features,
        out_features,
        *,
        config,
        qweight,
        scales,
        bias=None,
        padded_in_features=None,
        zero_points=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_config = config
        self.padded_in_features = padded_in_features or in_features
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)
        self.register_buffer("bias", bias)

    @classmethod
    def from_float(cls, module, config):
        if not isinstance(module, nn.Linear):
            raise TypeError("QuantizedLinear.from_float 仅接受 nn.Linear")

        quantized = quantize_weight(module.weight, config)
        bias = module.bias.detach().clone() if module.bias is not None else None
        return cls(
            module.in_features,
            module.out_features,
            config=config,
            qweight=quantized.qweight,
            scales=quantized.scales,
            bias=bias,
            padded_in_features=quantized.padded_in_features,
            zero_points=quantized.zero_points,
        )

    @property
    def quantized_weight(self):
        return QuantizedWeight(
            qweight=self.qweight,
            scales=self.scales,
            original_shape=(self.out_features, self.in_features),
            padded_in_features=self.padded_in_features,
            config=self.quantization_config,
            zero_points=self.zero_points,
        )

    @property
    def storage_bytes(self):
        return self.quantized_weight.storage_bytes + tensor_storage_bytes(self.bias)

    def forward(self, inputs):
        backend = get_backend(self.quantization_config.backend)
        return backend.linear(inputs, self.quantized_weight, self.bias)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mode={self.quantization_config.mode}, "
            f"backend={self.quantization_config.backend}"
        )
