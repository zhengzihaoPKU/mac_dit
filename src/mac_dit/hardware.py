import json
import platform
import subprocess

import torch

from .formatting import bytes_to_gb


class MPSUnavailableError(RuntimeError):
    """当前环境无法使用 PyTorch MPS。"""


def get_mps_status():
    """返回 MPS 是否可用及对应说明。"""
    if not torch.backends.mps.is_built():
        return False, "当前 PyTorch 未包含 MPS 支持，请安装支持 MPS 的 PyTorch。"
    if not torch.backends.mps.is_available():
        return False, "MPS 不可用，请检查 macOS 版本和 Apple Silicon 设备支持。"
    return True, "MPS 可用。"


def require_mps():
    """确保 MPS 可用，否则抛出包含原因的异常。"""
    available, message = get_mps_status()
    if not available:
        raise MPSUnavailableError(message)
    return message


def fetch_mps_memory():
    """获取当前进程的 MPS 内存数据，单位为 GB。"""
    if not torch.backends.mps.is_available():
        return None

    try:
        current_memory = torch.mps.current_allocated_memory()
        driver_memory = torch.mps.driver_allocated_memory()
    except (AttributeError, RuntimeError):
        return None

    memory = {
        "current_allocated_gb": bytes_to_gb(current_memory),
        "driver_allocated_gb": bytes_to_gb(driver_memory),
        "recommended_max_gb": None,
        "estimated_available_gb": None,
    }

    try:
        recommended_memory = torch.mps.recommended_max_memory()
    except (AttributeError, RuntimeError):
        return memory

    memory["recommended_max_gb"] = bytes_to_gb(recommended_memory)
    memory["estimated_available_gb"] = bytes_to_gb(
        max(recommended_memory - driver_memory, 0)
    )
    return memory


def print_mps_memory(memory=None):
    """输出当前进程的 MPS 内存数据。"""
    memory = memory or fetch_mps_memory()
    if not memory:
        print("MPS 内存信息: 无法获取")
        return

    print(f"MPS 当前张量内存: {memory['current_allocated_gb']:.2f} GB")
    print(f"Metal 驱动已分配内存: {memory['driver_allocated_gb']:.2f} GB")
    if memory["recommended_max_gb"] is not None:
        print(f"MPS 建议最大工作集: {memory['recommended_max_gb']:.2f} GB")
        print(f"MPS 估算可用内存: {memory['estimated_available_gb']:.2f} GB")


def _get_system_profile():
    """读取 macOS 硬件和 GPU 信息。"""
    try:
        result = subprocess.run(
            [
                "system_profiler",
                "SPHardwareDataType",
                "SPDisplaysDataType",
                "-json",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return json.loads(result.stdout)
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError):
        return {}


def _format_cpu_cores(cpu_description):
    """将 system_profiler 的 proc 8:4:4 转成易读格式。"""
    if not cpu_description:
        return "未知"

    parts = cpu_description.removeprefix("proc ").split(":")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        total, performance, efficiency = parts
        return f"{total}（{performance} 个性能核心 + {efficiency} 个能效核心）"
    return cpu_description


def fetch_mps_config():
    """获取 Mac 硬件、GPU 和 PyTorch MPS 配置。"""
    profile = _get_system_profile()
    hardware_items = profile.get("SPHardwareDataType", [])
    display_items = profile.get("SPDisplaysDataType", [])
    hardware = hardware_items[0] if hardware_items else {}
    mps_available, _ = get_mps_status()

    config = {
        "macos_version": platform.mac_ver()[0] or "未知",
        "machine_name": hardware.get("machine_name", "未知"),
        "machine_model": hardware.get("machine_model", "未知"),
        "chip": hardware.get("chip_type", "未知"),
        "cpu_cores": _format_cpu_cores(hardware.get("number_processors")),
        "unified_memory": hardware.get("physical_memory", "未知"),
        "gpus": [],
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps_available,
        "mps_memory": fetch_mps_memory() if mps_available else None,
    }

    for display in display_items:
        config["gpus"].append(
            {
                "model": display.get("sppci_model", display.get("_name", "未知")),
                "cores": display.get("sppci_cores", "未知"),
                "vram": display.get("spdisplays_vram")
                or display.get("spdisplays_vram_shared"),
                "metal_supported": (
                    display.get("spdisplays_metal") == "spdisplays_supported"
                ),
            }
        )

    return config


def print_mps_config(config=None):
    """以易读格式输出 Mac 和 MPS 配置信息。"""
    config = config or fetch_mps_config()

    print("=== Mac 硬件信息 ===")
    print(f"macOS 版本: {config['macos_version']}")
    print(f"电脑型号: {config['machine_name']} ({config['machine_model']})")
    print(f"芯片型号: {config['chip']}")
    print(f"CPU 核心: {config['cpu_cores']}")
    print(f"统一内存: {config['unified_memory']}")

    if config["gpus"]:
        for index, gpu in enumerate(config["gpus"], start=1):
            prefix = "GPU" if len(config["gpus"]) == 1 else f"GPU {index}"
            print(f"{prefix} 型号: {gpu['model']}")
            print(f"{prefix} 核心: {gpu['cores']}")
            if gpu["vram"]:
                print(f"{prefix} 显存: {gpu['vram']}")
            else:
                print(f"{prefix} 显存: 与 CPU 共享 {config['unified_memory']} 统一内存")
            print(f"{prefix} Metal 支持: {'是' if gpu['metal_supported'] else '否'}")
    else:
        print("GPU 信息: 无法获取")

    print(f"PyTorch MPS 已构建: {'是' if config['mps_built'] else '否'}")
    print(f"PyTorch MPS 可用: {'是' if config['mps_available'] else '否'}")
    if config["mps_memory"]:
        print_mps_memory(config["mps_memory"])
