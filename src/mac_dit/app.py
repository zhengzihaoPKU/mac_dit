"""统一命令行调度，并对 MLX 模块使用延迟导入。"""

import argparse
import sys


COMMAND_HELP = {
    "generate": "使用 PyTorch MPS 生成图片",
    "benchmark": "测试 PyTorch 量化性能",
    "check": "快速检查 MPS 是否可用",
    "hardware": "输出 Mac、GPU、内存和 MPS 信息",
    "model-info": "输出 DiT 结构、参数量和内存估算",
    "mlx-convert": "转换 MLX FP16、W4A16、W8A8 或 W4A4 checkpoint",
    "mlx-generate": "使用 MLX 生成图片",
}


def _root_parser():
    parser = argparse.ArgumentParser(
        prog="python src/main.py",
        description="Apple Silicon DiT 生成、量化和诊断工具",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("command", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _print_help(parser):
    parser.print_help()
    print("\n可用命令：")
    width = max(map(len, COMMAND_HELP))
    for command, description in COMMAND_HELP.items():
        print(f"  {command:<{width}}  {description}")
    print("\n查看子命令参数：python src/main.py <command> --help")


def _generate(argv):
    from .cli import main

    return main(argv)


def _benchmark(argv):
    from .benchmark import main

    return main(argv)


def _check(argv):
    from .hardware import MPSUnavailableError, require_mps

    argparse.ArgumentParser(description="检查 PyTorch MPS 是否可用").parse_args(
        argv
    )
    try:
        print(require_mps())
    except MPSUnavailableError as error:
        raise SystemExit(str(error)) from error


def _hardware(argv):
    from .hardware import print_mps_config

    parser = argparse.ArgumentParser(description="输出 Mac 硬件与 MPS 信息")
    parser.parse_args(argv)
    return print_mps_config()


def _model_info(argv):
    from .model_info import main

    return main(argv)


def _mlx_convert(argv):
    from .mlx_backend.cli import convert_main

    return convert_main(argv)


def _mlx_generate(argv):
    from .mlx_backend.cli import run_main

    return run_main(argv)


COMMANDS = {
    "generate": _generate,
    "benchmark": _benchmark,
    "check": _check,
    "hardware": _hardware,
    "model-info": _model_info,
    "mlx-convert": _mlx_convert,
    "mlx-generate": _mlx_generate,
}


def main(argv=None):
    """调度子命令；子命令参数原样传给对应的可复用模块。"""
    parser = _root_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.help or not args.command:
        _print_help(parser)
        return None
    handler = COMMANDS.get(args.command)
    if handler is None:
        parser.error(
            f"未知命令 {args.command!r}；可用命令："
            f"{', '.join(COMMANDS)}"
        )
    return handler(args.arguments)
