# mac_dit

在 Apple Silicon Mac 上使用 PyTorch MPS 和 Hugging Face Diffusers 运行
`facebook/DiT-XL-2-256`，生成一张 256×256 的虎斑猫图片。

## 环境要求

- 搭载 Apple Silicon 芯片的 Mac
- 支持 MPS 的 macOS 与 PyTorch 版本
- [Homebrew](https://brew.sh/)
- [uv](https://docs.astral.sh/uv/)
- 可访问 Hugging Face 以下载模型

## 项目结构

```text
mac_dit/
├── image/                  # 生成的图片
├── model/                  # Hugging Face 模型缓存
├── scripts/                # 环境安装、登录和检查脚本
├── src/
│   ├── check_backend.py    # 检查 MPS 是否可用
│   └── run_dit.py          # 加载模型并生成图片
├── .venv/                  # 本地 Python 虚拟环境
└── .gitignore
```

`model/` 和 `.venv/` 中只有说明文件会提交到 Git，模型权重、缓存和虚拟环境文件均会被忽略。

## 快速开始

请在项目根目录执行以下命令。

### 1. 安装 uv

```bash
brew install uv
```

也可以运行：

```bash
bash scripts/step1_uv_install.sh
```

### 2. 创建虚拟环境并安装依赖

```bash
uv venv
uv pip install torch diffusers transformers accelerate Pillow huggingface_hub
```

如需激活虚拟环境，请根据当前 Shell 选择命令：

```bash
# zsh 或 bash
source .venv/bin/activate

# fish
source .venv/bin/activate.fish
```

`scripts/step2_env_setup.sh` 使用 fish 的激活脚本，适合在 fish 中执行：

```fish
source scripts/step2_env_setup.sh
```

### 3. 登录 Hugging Face

```bash
uv run hf auth login
```

也可以运行：

```bash
bash scripts/step3_hf_login.sh
```

### 4. 检查 MPS

```bash
uv run python src/check_backend.py
```

也可以运行：

```bash
bash scripts/step4_check_backend.sh
```

### 5. 生成图片

```bash
uv run python src/run_dit.py
```

首次运行时会从 Hugging Face 下载模型，所需时间取决于网络速度。模型缓存在 `model/`，生成结果保存为：

```text
image/dit_generated_image.png
```

## 调整生成参数

可以在 `src/run_dit.py` 中修改以下参数：

- `class_labels`：ImageNet 类别编号，当前的 `281` 表示虎斑猫。
- `num_inference_steps`：推理步数，步数越多通常耗时越长。
- `output_path`：生成图片的保存位置。
- `model_id`：需要加载的 Hugging Face 模型 ID。

## 常见问题

### MPS 不可用

先执行后端检查脚本，并确认当前 Mac、macOS 和 PyTorch 均支持 MPS。

### 模型下载中断

重新运行生成命令即可。Hugging Face 会利用 `model/` 中已有的缓存继续下载。

### 重新创建虚拟环境

删除本地 `.venv/` 后，重新执行“创建虚拟环境并安装依赖”中的命令即可。虚拟环境不需要上传到 GitHub。
