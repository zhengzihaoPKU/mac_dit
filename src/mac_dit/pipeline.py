import time
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

from .config import GenerationConfig
from .hardware import print_mps_memory


@dataclass(frozen=True, slots=True)
class GenerationResult:
    image_path: Path
    elapsed_seconds: float


def load_pipeline(config=None, *, move_to_device=True):
    """加载 DiT pipeline，并按需移动到目标设备。"""
    config = config or GenerationConfig()
    torch_dtype = getattr(torch, config.dtype)

    print(f"正在使用 {config.device} 进行加速... 🚀")
    pipe = DiTPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch_dtype,
        cache_dir=config.cache_dir,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if move_to_device:
        pipe = pipe.to(config.device)
        if config.device == "mps":
            print_mps_memory()
    print("模型加载完成。")
    return pipe


def _synchronize(device):
    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def generate_image(pipe, config=None):
    """生成并保存图片，返回输出路径和纯推理耗时。"""
    config = config or GenerationConfig()

    _synchronize(config.device)
    start_time = time.perf_counter()
    image = pipe(
        class_labels=[config.class_label],
        num_inference_steps=config.inference_steps,
    ).images[0]
    _synchronize(config.device)
    elapsed_seconds = time.perf_counter() - start_time

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / f"dit_generated_image_{config.class_label}.png"
    image.save(output_path)
    return GenerationResult(output_path, elapsed_seconds)
