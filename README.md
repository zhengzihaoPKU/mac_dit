# mac_dit

**中文** | [English](README_EN.md)

在 Apple Silicon Mac 上使用 PyTorch MPS 和 Hugging Face Diffusers 运行
`facebook/DiT-XL-2-256`，根据 ImageNet 类别生成 256×256 图片。

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
│   ├── mac_dit/            # 可复用的核心模块
│   │   ├── cli.py          # 命令行参数和主流程
│   │   ├── config.py       # 统一配置和默认值
│   │   ├── formatting.py   # 内存与参数量格式化
│   │   ├── hardware.py     # Mac、GPU 和 MPS 信息
│   │   ├── model_info.py   # DiT 模型信息统计
│   │   ├── pipeline.py     # 模型加载、推理和图片保存
│   │   ├── benchmark.py    # 量化性能基准
│   │   ├── mlx_backend/    # MLX DiT、QMM 算子、转换和生成管线
│   │   └── quantization/   # 量化算法、模块、后端与序列化
│   ├── check_backend.py    # MPS 检查兼容入口
│   ├── benchmark_quantization.py # 量化基准入口
│   ├── convert_to_mlx.py   # Diffusers 权重转 MLX 入口
│   ├── run_mlx_dit.py      # MLX 低精度图片生成入口
│   ├── model_config_fetch.py # 模型信息兼容入口
│   ├── mps_config_fetch.py # Mac 硬件信息兼容入口
│   └── run_dit.py          # 图片生成兼容入口
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
uv pip install torch diffusers transformers accelerate Pillow huggingface_hub safetensors mlx
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

查看完整的 Mac、GPU 和 MPS 信息：

```bash
uv run python src/mps_config_fetch.py
```

查看 DiT 模型的参数量、数据类型和估算内存：

```bash
uv run python src/model_config_fetch.py
```

### 5. 生成图片

```bash
uv run python src/run_dit.py
```

也可以直接传入类别和推理步数，例如生成虎斑猫：

```bash
uv run python src/run_dit.py --class-label 281 --steps 25 --seed 42
```

首次运行时会从 Hugging Face 下载模型，所需时间取决于网络速度。模型缓存在 `model/`，生成结果保存为：

```text
image/dit_generated_image_<类别编号>_<精度>_seed<种子>.png
```

## 模型量化

项目提供 INT8 和 INT4 weight-only 量化。默认只量化 DiT Transformer 中的 attention 与 feed-forward Linear，AdaNorm、timestep embedding、最终输出投影和 VAE 保持 FP16。

### INT8 W8A16

```bash
uv run python src/run_dit.py \
  --class-label 281 \
  --seed 42 \
  --quantization int8
```

INT8 使用 per-output-channel 对称量化，激活保持 FP16。

### INT4 W4A16

```bash
uv run python src/run_dit.py \
  --class-label 281 \
  --seed 42 \
  --quantization int4 \
  --group-size 128
```

INT4 使用 group-wise 对称量化，并将两个 INT4 权重打包到一个字节。可以测试 `64` 或 `128` 的 group size。

### 调整量化范围

量化全部 Linear：

```bash
uv run python src/run_dit.py --quantization int8 --quantize-all-linears
```

按模块名称排除敏感层：

```bash
uv run python src/run_dit.py \
  --quantization int8 \
  --exclude-layer transformer_blocks.0 \
  --exclude-layer transformer_blocks.27
```

### 保存和加载量化模型

```bash
# 只量化并保存，不生成图片。默认保存到：
# model/quantized/DiT-XL-2-256-int4-g128/
uv run python src/run_dit.py \
  --quantization int4 \
  --group-size 128 \
  --quantize-only

# 也可以显式指定目录
uv run python src/run_dit.py \
  --quantization int4 \
  --quantize-only \
  --save-quantized ./model/quantized/my-dit-int4

# 直接加载已保存的量化 Transformer
uv run python src/run_dit.py \
  --load-quantized ./model/quantized/DiT-XL-2-256-int4-g128

# 使用另一个后端加载同一份量化权重
uv run python src/run_dit.py \
  --load-quantized ./model/quantized/DiT-XL-2-256-int4-g128 \
  --quant-backend metal
```

量化目录包含 `transformer.safetensors` 与 `quantization.json`。建议将其放在 `model/` 下，避免上传大型权重文件。

### 性能基准

使用相同类别、seed、scheduler 和推理步数分别运行 FP16、INT8 与 INT4：

```bash
# FP16
uv run python src/benchmark_quantization.py \
  --class-label 281 --seed 42 --steps 25 --warmup 1 --repeats 3

# INT8
uv run python src/benchmark_quantization.py \
  --class-label 281 --seed 42 --steps 25 \
  --quantization int8 --warmup 1 --repeats 3 \
  --json ./benchmark-int8.json

# INT4
uv run python src/benchmark_quantization.py \
  --class-label 281 --seed 42 --steps 25 \
  --quantization int4 --group-size 128 --warmup 1 --repeats 3
```

当前 `reference` 后端在每个 Linear 前向时反量化，再调用浮点 `F.linear`。它用于验证量化正确性和内存压缩，不保证比 FP16 更快。

也可以启用实验性的 Metal 后端：

```bash
uv run python src/run_dit.py \
  --quantization int4 --quant-backend metal
```

该后端通过 `torch.mps.compile_shader` 调用自定义 W8A16/W4A16 Linear。算子直接读取 INT8 或打包 INT4 权重，在乘法循环中应用 scale，不会先生成完整 FP16 权重矩阵；激活保持 FP16，累加使用 FP32。当前实现是正确性优先的基础 kernel，尚未加入矩阵分块、SIMD 协作和线程组缓存等性能优化，需要支持 `torch.mps.compile_shader` 的 PyTorch 和可用的 MPS 环境。环境不满足条件时，请使用默认的 `reference` 后端。

## MLX 加速后端

MLX 后端是推荐的低比特加速路径。它不会逐层在 PyTorch 与 MLX 之间转换数据，而是让完整的 28 层 DiT Transformer 留在 MLX/Metal 中执行。只有每个扩散步骤结束后，尺寸很小的 latent 会交给 CPU scheduler；VAE 最终只解码一次。

### 1. 转换并量化权重

```bash
uv run python src/convert_to_mlx.py --bits 4 --group-size 128
```

默认生成：

```text
model/mlx/DiT-XL-2-256-w4-g128/
├── dit.safetensors  # MLX FP16 与打包 INT4 参数
└── mlx_config.json  # 网络结构、量化参数和量化层列表
```

默认量化 168 个 attention/FFN Linear。也可以使用 `--bits 8`，或者通过 `--output-dir` 指定其他目录。转换只需执行一次。

### 权重和激活同时量化

项目支持 MLX 原生 `mx.qqmm`：权重在 checkpoint 中以低精度保存，激活在每次 Linear 计算时动态量化，然后由融合的低精度 Metal 算子完成矩阵乘法。DiT 中的 bias 在 `qqmm` 之后以浮点相加。

MXFP8 W8A8（固定 group size 32）：

```bash
uv run python src/convert_to_mlx.py --activation-quantization mxfp8
uv run python src/run_mlx_dit.py \
  --mlx-model-dir model/mlx/DiT-XL-2-256-mxfp8-w8a8 \
  --class-label 281 --steps 25 --seed 42
```

NVFP4 W4A4（固定 group size 16）：

```bash
uv run python src/convert_to_mlx.py --activation-quantization nvfp4
uv run python src/run_mlx_dit.py \
  --mlx-model-dir model/mlx/DiT-XL-2-256-nvfp4-w4a4 \
  --class-label 281 --steps 25 --seed 42
```

两种模式默认都量化 168 个 attention/FFN Linear，AdaNorm、embedding、最终投影和 VAE 保持浮点。可以继续使用 `--quantize-all-linears` 或 `--exclude-layer` 调整范围。

### 2. 使用 MLX 生成图片

```bash
uv run python src/run_mlx_dit.py \
  --class-label 281 \
  --steps 25 \
  --seed 42
```

默认启用 `mx.compile`。首次运行需要编译计算图，后续运行会更快；排查问题时可添加 `--no-compile`。

在本项目的 M3 MacBook Air（10 核 GPU、16 GB 内存）上，以类别 281、seed 42、25 步进行同口径测试：

- PyTorch MPS FP16：总推理约 6.02 秒。
- MLX W4A16 group-128：DiT 去噪约 5.01 秒，总推理约 5.45 秒，MLX 峰值约 0.99 GB。
- MLX MXFP8 W8A8：DiT 去噪约 18.00 秒，总推理约 18.52 秒，MLX 峰值约 1.21 GB。
- MLX NVFP4 W4A4：DiT 去噪约 17.09 秒，总推理约 17.49 秒，MLX 峰值约 1.01 GB。
- MLX 只量化 FFN：总推理约 5.55 秒，因此默认仍量化 attention 和 FFN。

这台 M3 上的 `mx.qqmm` 动态激活量化没有带来加速，开销反而使其约慢 3.2–3.4 倍，因此默认仍是 W4A16。MXFP8/NVFP4 作为可选实验路径，适合在具有相应低精度硬件吞吐的新设备上重新基准测试。单次运行会受首次编译、温度和系统负载影响，请使用多次 warmup 后的中位数评价最终性能。

### 代码接口

MLX 代码按职责拆分：

- `mlx_backend/operators.py`：真正调用 `mx.quantized_matmul`、`mx.qqmm` 和 fused attention 的算子层。
- `mlx_backend/model.py`：PatchEmbed、AdaLayerNormZero、Attention、FFN 和完整 DiT 网络。
- `mlx_backend/conversion.py`：PyTorch OIHW→MLX OHWI 转置、参数重命名和 checkpoint 保存/加载。
- `mlx_backend/pipeline.py`：classifier-free guidance、scheduler 桥接、VAE 解码和计时。
- `mlx_backend/cli.py`：转换与推理命令行接口。

核心量化算子接口：

```python
from mac_dit.mlx_backend.operators import quantized_matmul

output = quantized_matmul(
    inputs,
    packed_weight,
    scales,
    quantization_biases,
    bits=4,
    group_size=128,
)
```

权重与激活同时量化的算子接口：

```python
from mac_dit.mlx_backend.operators import quantized_quantized_matmul

output = quantized_quantized_matmul(
    inputs,
    packed_weight,
    scales,
    bits=4,
    group_size=16,
    mode="nvfp4",
)
```

完整模型接口：

```python
noise_and_sigma = model(
    sample,        # [B, 4, 32, 32]，NCHW FP16
    timesteps,     # [B]
    class_labels,  # [B]，空类别为 1000
)
```

加载已有 checkpoint：

```python
from mac_dit.mlx_backend.conversion import load_mlx_transformer

model, manifest = load_mlx_transformer(
    "model/mlx/DiT-XL-2-256-w4-g128"
)
```

## 调整生成参数

运行 `src/run_dit.py --help` 可以查看全部参数。常用参数包括：

- `--class-label`：ImageNet 类别编号，`281` 表示虎斑猫。
- `--steps`：推理步数，步数越多通常耗时越长。
- `--seed`：随机种子，用于复现实验和比较量化结果。
- `--output-dir`：生成图片的保存目录。
- `--model-id`：需要加载的 Hugging Face 模型 ID。
- `--cache-dir`：模型缓存目录。

需要在 `src` 下的 Python 代码中复用时，可以组合统一配置和核心函数：

```python
from mac_dit.config import GenerationConfig
from mac_dit.pipeline import generate_image, load_pipeline
from mac_dit.quantization import QuantizationConfig

config = GenerationConfig(
    class_label=281,
    inference_steps=25,
    seed=42,
    quantization=QuantizationConfig(mode="int8"),
)
pipe = load_pipeline(config)
result = generate_image(pipe, config)
print(result.image_path, result.elapsed_seconds)
```

## 常见问题

### MPS 不可用

先执行后端检查脚本，并确认当前 Mac、macOS 和 PyTorch 均支持 MPS。

### 模型下载中断

重新运行生成命令即可。Hugging Face 会利用 `model/` 中已有的缓存继续下载。

### 重新创建虚拟环境

删除本地 `.venv/` 后，重新执行“创建虚拟环境并安装依赖”中的命令即可。
