from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = "facebook/DiT-XL-2-256"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "model"
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

    def __post_init__(self):
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

        if self.inference_steps <= 0:
            raise ValueError("推理步数必须大于 0")
