import argparse
from pathlib import Path

from ..config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, DEFAULT_OUTPUT_DIR
from .config import MlxQuantizationConfig
from .conversion import convert_pretrained
from .pipeline import MlxGenerationConfig, generate_image


def default_mlx_directory(cache_dir, *, bits=4, group_size=128):
    return Path(cache_dir) / "mlx" / f"DiT-XL-2-256-w{bits}-g{group_size}"


def build_convert_parser():
    parser = argparse.ArgumentParser(description="将 Diffusers DiT 转换为 MLX")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir")
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--quantize-all-linears", action="store_true")
    parser.add_argument("--exclude-layer", action="append", default=[])
    return parser


def convert_main(argv=None):
    args = build_convert_parser().parse_args(argv)
    quantization = None
    if not args.no_quantization:
        quantization = MlxQuantizationConfig(
            bits=args.bits,
            group_size=args.group_size,
            include_patterns=() if args.quantize_all_linears else (".attn1.", ".ff."),
            exclude_patterns=tuple(args.exclude_layer),
        )
    output_dir = args.output_dir or default_mlx_directory(
        args.cache_dir,
        bits=args.bits,
        group_size=args.group_size,
    )
    if args.no_quantization and args.output_dir is None:
        output_dir = Path(args.cache_dir) / "mlx" / "DiT-XL-2-256-fp16"
    _, manifest = convert_pretrained(
        args.model_id,
        args.cache_dir,
        output_dir,
        quantization=quantization,
    )
    print(f"MLX checkpoint 已保存到：{output_dir}")
    print(f"量化 Linear：{len(manifest['quantized_layers'])} 层")
    return Path(output_dir)


def build_run_parser():
    parser = argparse.ArgumentParser(description="使用 MLX 量化 DiT 生成图片")
    parser.add_argument(
        "--mlx-model-dir",
        default=default_mlx_directory(DEFAULT_CACHE_DIR),
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--class-label", type=int, default=150)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--no-compile", action="store_true")
    return parser


def run_main(argv=None):
    args = build_run_parser().parse_args(argv)
    config = MlxGenerationConfig(
        mlx_model_dir=args.mlx_model_dir,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        class_label=args.class_label,
        inference_steps=args.steps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        compile_model=not args.no_compile,
    )
    result = generate_image(config)
    print(f"MLX DiT 去噪耗时：{result.denoising_seconds:.2f} 秒")
    print(f"MLX 总推理耗时：{result.elapsed_seconds:.2f} 秒")
    print(f"MLX 峰值内存：{result.peak_memory_gb:.2f} GB")
    print(f"图片已保存到：{result.image_path}")
    return result
