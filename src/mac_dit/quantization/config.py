from dataclasses import dataclass
from typing import Literal


QuantizationMode = Literal["int8", "int4"]
QuantizationBackend = Literal["reference", "metal"]

DEFAULT_INCLUDE_PATTERNS = (".attn1.", ".ff.")


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """Weight-only 量化配置。"""

    mode: QuantizationMode = "int8"
    backend: QuantizationBackend = "reference"
    group_size: int = 128
    include_patterns: tuple[str, ...] = DEFAULT_INCLUDE_PATTERNS
    exclude_patterns: tuple[str, ...] = ()

    def __post_init__(self):
        if self.mode not in ("int8", "int4"):
            raise ValueError(f"不支持的量化模式: {self.mode}")
        if self.backend not in ("reference", "metal"):
            raise ValueError(f"不支持的量化后端: {self.backend}")
        if self.group_size <= 0:
            raise ValueError("group_size 必须大于 0")
        if self.mode == "int4" and self.group_size % 2 != 0:
            raise ValueError("INT4 group_size 必须是偶数")

    @property
    def bits(self):
        return 8 if self.mode == "int8" else 4

    @property
    def label(self):
        return f"w{self.bits}a16-{self.backend}"
