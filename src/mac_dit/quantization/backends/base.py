from typing import Protocol

import torch

from ..types import QuantizedWeight


class QuantizationBackend(Protocol):
    name: str

    def linear(
        self,
        inputs: torch.Tensor,
        weight: QuantizedWeight,
        bias: torch.Tensor | None,
    ) -> torch.Tensor: ...

    def supports(self, weight: QuantizedWeight, device: torch.device) -> bool: ...
