import time
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers import DiTPipeline, DPMSolverMultistepScheduler

from .config import GenerationConfig
from .formatting import bytes_to_gb
from .hardware import print_mps_memory
from .quantization.serialization import load_quantized_model, save_quantized_model
from .quantization.transform import quantize_model


@dataclass(frozen=True, slots=True)
class GenerationResult:
    image_path: Path
    elapsed_seconds: float


def load_pipeline(config=None, *, move_to_device=True):
    """加载 DiT pipeline，并按需移动到目标设备。"""
    config = config or GenerationConfig()
    torch_dtype = getattr(torch, config.dtype)

    if move_to_device:
        print(f"正在使用 {config.device} 进行加速... 🚀")
    else:
        print("正在 CPU 上加载并处理模型权重...")
    pipe = DiTPipeline.from_pretrained(
        config.model_id,
        dtype=torch_dtype,
        cache_dir=config.cache_dir,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    report = None
    if config.load_quantized_dir:
        report = load_quantized_model(
            pipe.transformer,
            config.load_quantized_dir,
            backend=config.load_quantized_backend,
        )
    elif config.quantization:
        report = quantize_model(pipe.transformer, config.quantization)
        if config.save_quantized_dir:
            saved_path = save_quantized_model(
                pipe.transformer,
                config.save_quantized_dir,
                model_id=config.model_id,
            )
            print(f"量化 checkpoint 已保存到：{saved_path}")

    if report:
        print(
            f"已量化 {report.quantized_layers}/{report.total_linear_layers} 个 Linear，"
            f"所选层从 {bytes_to_gb(report.original_bytes):.2f} GB 降至 "
            f"{bytes_to_gb(report.quantized_bytes):.2f} GB "
            f"({report.compression_ratio:.2f}x)"
        )
        pipe._mac_dit_quantization_report = report

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
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    _synchronize(config.device)
    start_time = time.perf_counter()
    image = pipe(
        class_labels=[config.class_label],
        generator=generator,
        num_inference_steps=config.inference_steps,
    ).images[0]
    _synchronize(config.device)
    elapsed_seconds = time.perf_counter() - start_time

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / (
        f"dit_generated_image_{config.class_label}_"
        f"{config.precision_label}_seed{config.seed}.png"
    )
    image.save(output_path)
    return GenerationResult(output_path, elapsed_seconds)
