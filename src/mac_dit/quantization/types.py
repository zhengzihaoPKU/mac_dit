from dataclasses import dataclass

import torch

from .config import QuantizationConfig


def tensor_storage_bytes(tensor):
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True, slots=True)
class QuantizedWeight:
    """与具体执行后端无关的量化权重。"""

    qweight: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]
    padded_in_features: int
    config: QuantizationConfig
    zero_points: torch.Tensor | None = None

    @property
    def storage_bytes(self):
        return sum(
            tensor_storage_bytes(tensor)
            for tensor in (self.qweight, self.scales, self.zero_points)
        )
