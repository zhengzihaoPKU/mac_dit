from dataclasses import dataclass
from pathlib import Path

from .quantization.config import QuantizationConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "facebook/DiT-XL-2-256"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "model"
DEFAULT_QUANTIZED_ROOT = DEFAULT_CACHE_DIR / "quantized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "image"
DEFAULT_DEVICE = "mps"
DEFAULT_DTYPE = "float16"
DEFAULT_CLASS_LABEL = 150
DEFAULT_INFERENCE_STEPS = 25


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """模型加载和图片生成所需的统一配置。"""

    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Path = DEFAULT_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    class_label: int = DEFAULT_CLASS_LABEL
    inference_steps: int = DEFAULT_INFERENCE_STEPS
    seed: int = 0
    quantization: QuantizationConfig | None = None
    load_quantized_dir: Path | None = None
    load_quantized_backend: str | None = None
    save_quantized_dir: Path | None = None

    def __post_init__(self):
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.load_quantized_dir is not None:
            object.__setattr__(
                self,
                "load_quantized_dir",
                Path(self.load_quantized_dir),
            )
        if self.save_quantized_dir is not None:
            object.__setattr__(
                self,
                "save_quantized_dir",
                Path(self.save_quantized_dir),
            )

        if self.inference_steps <= 0:
            raise ValueError("推理步数必须大于 0")
        if self.load_quantized_dir and self.quantization:
            raise ValueError("不能同时加载量化 checkpoint 和重新量化模型")
        if self.load_quantized_backend and not self.load_quantized_dir:
            raise ValueError("只有加载量化 checkpoint 时才能覆盖量化后端")
        if self.save_quantized_dir and not self.quantization:
            raise ValueError("只有启用量化时才能保存量化 checkpoint")

    @property
    def precision_label(self):
        if self.quantization:
            return self.quantization.label
        if self.load_quantized_dir:
            return "quantized"
        return self.dtype
