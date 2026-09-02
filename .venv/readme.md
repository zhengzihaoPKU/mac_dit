# Python 虚拟环境目录

此目录用于存放由 `uv` 创建的项目虚拟环境及已安装的 Python 依赖。

除本说明文件外，虚拟环境中的可执行文件、第三方包和配置均已被 `.gitignore` 忽略，不会上传到 GitHub。克隆项目后，请在项目根目录重新创建环境：

```bash
uv venv
uv pip install torch diffusers transformers accelerate Pillow huggingface_hub safetensors
```

如需激活环境，请根据当前 Shell 选择命令：

```bash
# zsh 或 bash
source .venv/bin/activate

# fish
source .venv/bin/activate.fish
```

也可以不手动激活，直接通过 `uv run` 执行项目命令：

```bash
uv run python src/run_dit.py
```

虚拟环境出现异常时，可以删除 `.venv/` 并按上述步骤重新创建；不要复制或提交其他机器生成的虚拟环境。
