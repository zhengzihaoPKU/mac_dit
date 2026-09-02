import argparse
import re
from pathlib import Path

from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CLASS_LABEL,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUANTIZED_ROOT,
    GenerationConfig,
)
from .hardware import MPSUnavailableError, require_mps
from .pipeline import generate_image, load_pipeline
from .quantization.backends import available_backends
from .quantization.config import DEFAULT_INCLUDE_PATTERNS, QuantizationConfig


AUTO_QUANTIZED_DIR = object()


def default_quantized_dir(model_id, quantization, *, cache_dir=DEFAULT_QUANTIZED_ROOT):
    """生成位于 model/quantized 下、可读且稳定的量化目录名。"""
    model_name = model_id.rstrip("/").rsplit("/", 1)[-1]
    model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name)
    suffix = quantization.mode
    if quantization.mode == "int4":
        suffix += f"-g{quantization.group_size}"
    root = Path(cache_dir)
    return root / f"{model_name}-{suffix}"


def add_generation_arguments(parser):
    parser.add_argument("--class-label", type=int, default=DEFAULT_CLASS_LABEL)
    parser.add_argument("--steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    quantization = parser.add_mutually_exclusive_group()
    quantization.add_argument(
        "--quantization",
        choices=("none", "int8", "int4"),
        default="none",
        help="Weight-only 量化模式",
    )
    quantization.add_argument(
        "--load-quantized",
        metavar="DIR",
        help="加载之前保存的量化 Transformer",
    )
    parser.add_argument(
        "--quant-backend",
        choices=available_backends(),
        help="执行后端；新量化默认 reference，加载 checkpoint 时默认沿用原配置",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--quantize-all-linears",
        action="store_true",
        help="量化全部 Linear；默认只量化 attention 和 feed-forward",
    )
    parser.add_argument(
        "--exclude-layer",
        action="append",
        default=[],
        help="按名称排除包含该字符串的层，可重复传入",
    )
    parser.add_argument(
        "--save-quantized",
        nargs="?",
        const=AUTO_QUANTIZED_DIR,
        metavar="DIR",
        help="保存量化 Transformer；省略 DIR 时保存到 model/quantized 的独立目录",
    )
    return parser


def build_parser():
    parser = argparse.ArgumentParser(description="使用 DiT 在 Apple MPS 上生成图片")
    add_generation_arguments(parser)
    parser.add_argument(
        "--quantize-only",
        action="store_true",
        help="只量化并保存权重，不执行 MPS 图片生成",
    )
    return parser


def generation_config_from_args(args):
    quantization = None
    if args.quantization != "none":
        quantization = QuantizationConfig(
            mode=args.quantization,
            backend=args.quant_backend or "reference",
            group_size=args.group_size,
            include_patterns=() if args.quantize_all_linears else DEFAULT_INCLUDE_PATTERNS,
            exclude_patterns=tuple(args.exclude_layer),
        )

    save_quantized_dir = args.save_quantized
    if getattr(args, "quantize_only", False):
        if quantization is None:
            raise ValueError("--quantize-only 需要同时指定 INT8 或 INT4 量化")
        if save_quantized_dir is None:
            save_quantized_dir = AUTO_QUANTIZED_DIR
    if save_quantized_dir is AUTO_QUANTIZED_DIR:
        if quantization is None:
            raise ValueError("--save-quantized 需要同时指定 INT8 或 INT4 量化")
        save_quantized_dir = default_quantized_dir(
            args.model_id,
            quantization,
            cache_dir=Path(args.cache_dir) / "quantized",
        )

    return GenerationConfig(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        class_label=args.class_label,
        inference_steps=args.steps,
        seed=args.seed,
        quantization=quantization,
        load_quantized_dir=args.load_quantized,
        load_quantized_backend=args.quant_backend if args.load_quantized else None,
        save_quantized_dir=save_quantized_dir,
    )


def ensure_mps():
    try:
        print(require_mps())
    except MPSUnavailableError as error:
        raise SystemExit(str(error)) from error


def run_generation(config):
    ensure_mps()
    pipe = load_pipeline(config)
    print("开始生成图片...")
    result = generate_image(pipe, config)
    print(f"图片生成耗时: {result.elapsed_seconds:.2f} 秒")
    print(f"图片生成完成！🎉 已经保存到 {result.image_path}")
    return result


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = generation_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.quantize_only:
        load_pipeline(config, move_to_device=False)
        print(f"量化权重保存完成：{config.save_quantized_dir}")
        return config.save_quantized_dir
    return run_generation(config)
