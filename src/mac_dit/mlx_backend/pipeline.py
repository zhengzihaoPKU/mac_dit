"""MLX DiT 生成管线。

DiT 的全部 28 层留在 MLX/Metal 中执行。Diffusers scheduler 在 CPU 上处理很小的
latent；每个扩散步骤只跨框架传递一次 latent，不会逐 Linear 切换框架。
"""

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor

from ..config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, DEFAULT_OUTPUT_DIR
from .conversion import load_mlx_transformer


@dataclass(frozen=True, slots=True)
class MlxGenerationConfig:
    mlx_model_dir: Path
    model_id: str = DEFAULT_MODEL_ID
    cache_dir: Path = DEFAULT_CACHE_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    class_label: int = 150
    inference_steps: int = 25
    seed: int = 0
    guidance_scale: float = 4.0
    compile_model: bool = True

    def __post_init__(self):
        for field in ("mlx_model_dir", "cache_dir", "output_dir"):
            object.__setattr__(self, field, Path(getattr(self, field)))
        if self.inference_steps <= 0:
            raise ValueError("推理步数必须大于 0")
        if not 0 <= self.class_label <= 999:
            raise ValueError("ImageNet 类别编号必须在 0 到 999 之间")


@dataclass(frozen=True, slots=True)
class MlxGenerationResult:
    image_path: Path
    elapsed_seconds: float
    denoising_seconds: float
    peak_memory_gb: float


def _torch_to_mlx(tensor, *, dtype=mx.float16):
    return mx.array(tensor.detach().cpu().contiguous().numpy()).astype(dtype)


def _mlx_to_torch(array, *, dtype=torch.float16):
    mx.eval(array)
    return torch.from_numpy(np.asarray(array)).to(dtype=dtype)


def load_runtime(config):
    """加载 MLX DiT、Diffusers scheduler 和 VAE。"""
    model, manifest = load_mlx_transformer(config.mlx_model_dir)
    forward = model
    if config.compile_model:
        forward = mx.compile(
            lambda sample, timesteps, labels: model(sample, timesteps, labels)
        )

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        config.model_id,
        subfolder="scheduler",
        cache_dir=config.cache_dir,
    )

    vae_device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae_dtype = torch.float16 if vae_device == "mps" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        config.model_id,
        subfolder="vae",
        cache_dir=config.cache_dir,
        dtype=vae_dtype,
    ).to(vae_device)
    vae.eval()
    return forward, manifest, scheduler, vae, vae_device


@torch.no_grad()
def generate_image(config):
    """使用 MLX 量化 DiT 生成图片并返回耗时与 MLX 峰值内存。"""
    forward, manifest, scheduler, vae, vae_device = load_runtime(config)
    model_config = manifest["model"]
    batch_size = 1
    latent_channels = model_config["in_channels"]
    latent_size = model_config["sample_size"]

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    latents = torch.randn(
        (batch_size, latent_channels, latent_size, latent_size),
        generator=generator,
        dtype=torch.float16,
    )
    if config.guidance_scale > 1:
        latent_model_input = torch.cat([latents, latents], dim=0)
        class_labels = torch.tensor([config.class_label, 1000], dtype=torch.int32)
    else:
        latent_model_input = latents
        class_labels = torch.tensor([config.class_label], dtype=torch.int32)

    scheduler.set_timesteps(config.inference_steps)
    mx.reset_peak_memory()
    inference_started = time.perf_counter()

    for timestep in scheduler.timesteps:
        if config.guidance_scale > 1:
            half = latent_model_input[: latent_model_input.shape[0] // 2]
            latent_model_input = torch.cat([half, half], dim=0)
        latent_model_input = scheduler.scale_model_input(
            latent_model_input,
            timestep,
        )

        batch = latent_model_input.shape[0]
        mlx_sample = _torch_to_mlx(latent_model_input)
        mlx_timesteps = mx.full((batch,), int(timestep.item()), dtype=mx.int32)
        mlx_labels = mx.array(class_labels.numpy())
        noise_pred = _mlx_to_torch(
            forward(mlx_sample, mlx_timesteps, mlx_labels)
        )

        if config.guidance_scale > 1:
            epsilon = noise_pred[:, :latent_channels]
            rest = noise_pred[:, latent_channels:]
            conditional, unconditional = epsilon.chunk(2, dim=0)
            guided = unconditional + config.guidance_scale * (
                conditional - unconditional
            )
            epsilon = torch.cat([guided, guided], dim=0)
            noise_pred = torch.cat([epsilon, rest], dim=1)

        if model_config["out_channels"] // 2 == latent_channels:
            noise_pred = noise_pred[:, :latent_channels]
        latent_model_input = scheduler.step(
            noise_pred,
            timestep,
            latent_model_input,
        ).prev_sample

    mx.synchronize()
    denoising_seconds = time.perf_counter() - inference_started
    peak_memory_gb = mx.get_peak_memory() / (1024**3)

    latents = (
        latent_model_input.chunk(2, dim=0)[0]
        if config.guidance_scale > 1
        else latent_model_input
    )
    latents = latents.to(device=vae_device, dtype=vae.dtype)
    latents = latents / vae.config.scaling_factor
    images = vae.decode(latents).sample
    images = VaeImageProcessor().postprocess(images, output_type="pil")
    elapsed_seconds = time.perf_counter() - inference_started

    config.output_dir.mkdir(parents=True, exist_ok=True)
    quantization = manifest.get("quantization")
    precision = (
        f"mlx-w{quantization['bits']}a16-g{quantization['group_size']}"
        f"-q{len(manifest.get('quantized_layers', ()))}"
        if quantization
        else "mlx-fp16"
    )
    output_path = config.output_dir / (
        f"dit_generated_image_{config.class_label}_{precision}_seed{config.seed}.png"
    )
    images[0].save(output_path)
    return MlxGenerationResult(
        output_path,
        elapsed_seconds,
        denoising_seconds,
        peak_memory_gb,
    )
