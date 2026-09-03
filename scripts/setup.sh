#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    brew install uv
fi

uv venv
uv pip install \
    torch \
    diffusers \
    transformers \
    accelerate \
    Pillow \
    huggingface_hub \
    safetensors \
    mlx

echo "环境安装完成。运行 uv run python src/main.py --help 查看命令。"
