# 模型缓存目录

运行 `src/run_dit.py` 时，Hugging Face Diffusers 会将
`facebook/DiT-XL-2-256` 的配置和模型权重自动下载到此目录。

无需手动放置模型文件。首次下载可能耗时较长，并占用数 GB 磁盘空间；后续运行会优先复用本地缓存。

除本说明文件外，此目录中的模型权重、锁文件和缓存文件均已被 `.gitignore` 忽略，不会上传到 GitHub。克隆项目后，运行以下命令即可重新下载所需模型：

```bash
uv run python src/run_dit.py
```

请从项目根目录执行该命令，以确保模型缓存到正确位置。
