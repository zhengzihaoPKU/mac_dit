import argparse

from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CLASS_LABEL,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    GenerationConfig,
)
from .hardware import MPSUnavailableError, require_mps
from .pipeline import generate_image, load_pipeline


def build_parser():
    parser = argparse.ArgumentParser(description="使用 DiT 在 Apple MPS 上生成图片")
    parser.add_argument("--class-label", type=int, default=DEFAULT_CLASS_LABEL)
    parser.add_argument("--steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = GenerationConfig(
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        class_label=args.class_label,
        inference_steps=args.steps,
    )

    try:
        print(require_mps())
    except MPSUnavailableError as error:
        raise SystemExit(str(error)) from error

    pipe = load_pipeline(config)
    print("开始生成图片...")
    result = generate_image(pipe, config)
    print(f"图片生成耗时: {result.elapsed_seconds:.2f} 秒")
    print(f"图片生成完成！🎉 已经保存到 {result.image_path}")
