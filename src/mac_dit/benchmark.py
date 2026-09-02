import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .cli import add_generation_arguments, ensure_mps, generation_config_from_args
from .hardware import fetch_mps_memory
from .pipeline import generate_image, load_pipeline


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    precision: str
    class_label: int
    seed: int
    inference_steps: int
    load_seconds: float
    latencies_seconds: tuple[float, ...]
    median_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    memory_after_load: dict | None
    memory_after_generation: dict | None


def benchmark(config, *, warmup=0, repeats=3):
    if warmup < 0:
        raise ValueError("warmup 不能小于 0")
    if repeats <= 0:
        raise ValueError("repeats 必须大于 0")

    load_started = time.perf_counter()
    pipe = load_pipeline(config)
    load_seconds = time.perf_counter() - load_started
    memory_after_load = fetch_mps_memory()

    for _ in range(warmup):
        generate_image(pipe, config)

    latencies = tuple(
        generate_image(pipe, config).elapsed_seconds for _ in range(repeats)
    )
    return BenchmarkResult(
        precision=config.precision_label,
        class_label=config.class_label,
        seed=config.seed,
        inference_steps=config.inference_steps,
        load_seconds=load_seconds,
        latencies_seconds=latencies,
        median_seconds=statistics.median(latencies),
        minimum_seconds=min(latencies),
        maximum_seconds=max(latencies),
        memory_after_load=memory_after_load,
        memory_after_generation=fetch_mps_memory(),
    )


def build_parser():
    parser = argparse.ArgumentParser(description="运行 DiT 量化性能基准")
    add_generation_arguments(parser)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", metavar="PATH", help="将结果写入 JSON 文件")
    return parser


def print_result(result):
    print("=== DiT 量化基准 ===")
    print(f"精度: {result.precision}")
    print(f"模型加载与量化: {result.load_seconds:.2f} 秒")
    print(f"推理延迟: {', '.join(f'{value:.2f}' for value in result.latencies_seconds)} 秒")
    print(f"推理延迟中位数: {result.median_seconds:.2f} 秒")
    if result.memory_after_generation:
        memory = result.memory_after_generation
        print(f"MPS 当前张量内存: {memory['current_allocated_gb']:.2f} GB")
        print(f"Metal 驱动已分配内存: {memory['driver_allocated_gb']:.2f} GB")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = generation_config_from_args(args)
        ensure_mps()
        result = benchmark(config, warmup=args.warmup, repeats=args.repeats)
    except ValueError as error:
        parser.error(str(error))

    print_result(result)
    if args.json:
        output_path = Path(args.json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
