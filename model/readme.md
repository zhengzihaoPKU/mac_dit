# 模型缓存目录

运行 `src/main.py generate` 时，Hugging Face Diffusers 会将
`facebook/DiT-XL-2-256` 的配置和模型权重自动下载到此目录。

无需手动放置模型文件。首次下载可能耗时较长，并占用数 GB 磁盘空间；后续运行会优先复用本地缓存。

除本说明文件外，此目录中的模型权重、锁文件和缓存文件均已被 `.gitignore` 忽略，不会上传到 GitHub。克隆项目后，运行以下命令即可重新下载所需模型：

```bash
uv run python src/main.py generate
```

请从项目根目录执行该命令，以确保模型缓存到正确位置。

INT8/INT4 Transformer 会保存在 `model/quantized/` 的独立目录中，例如：

```bash
uv run python src/main.py generate \
  --quantization int4 \
  --group-size 128 \
  --quantize-only
```

上面的命令默认写入 `model/quantized/DiT-XL-2-256-int4-g128/`。量化 checkpoint 同样不会上传到 GitHub。

MLX 转换后的权重保存在独立的 `model/mlx/` 目录：

```bash
uv run python src/main.py mlx-convert --bits 4 --group-size 128
```

默认输出为 `model/mlx/DiT-XL-2-256-w4-g128/`，其中包含 `dit.safetensors` 和 `mlx_config.json`。

如需同时量化权重和激活，可以生成独立的 MXFP8 W8A8 或 NVFP4 W4A4 checkpoint：

```bash
uv run python src/main.py mlx-convert --activation-quantization mxfp8
uv run python src/main.py mlx-convert --activation-quantization nvfp4
```

对应目录为 `model/mlx/DiT-XL-2-256-mxfp8-w8a8/` 和 `model/mlx/DiT-XL-2-256-nvfp4-w4a4/`。这些大型 checkpoint 同样不会上传到 GitHub。
