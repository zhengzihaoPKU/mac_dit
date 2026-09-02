import torch
import torch.nn.functional as F

from ..algorithms import dequantize_weight


class ReferenceBackend:
    """以浮点反量化加 F.linear 实现的正确性参考后端。"""

    name = "reference"

    def supports(self, weight, device):
        return device.type in ("cpu", "mps") and weight.config.mode in ("int8", "int4")

    def linear(self, inputs, weight, bias):
        if not self.supports(weight, inputs.device):
            raise RuntimeError(
                f"Reference 后端不支持 {inputs.device.type}/{weight.config.mode}"
            )
        dequantized = dequantize_weight(weight, dtype=inputs.dtype)
        return F.linear(inputs, dequantized, bias)
