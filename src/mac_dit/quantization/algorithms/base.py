from typing import Protocol

import torch

from ..config import QuantizationConfig
from ..types import QuantizedWeight


class WeightQuantizer(Protocol):
    def __call__(
        self,
        weight: torch.Tensor,
        config: QuantizationConfig,
    ) -> QuantizedWeight: ...
