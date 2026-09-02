# mac_dit

[中文](README.md) | **English**

Run `facebook/DiT-XL-2-256` on Apple Silicon with PyTorch MPS, Hugging Face Diffusers, and MLX to generate 256×256 images from ImageNet class labels.

## Requirements

- A Mac with Apple Silicon
- A macOS and PyTorch version with MPS support
- [Homebrew](https://brew.sh/)
- [uv](https://docs.astral.sh/uv/)
- Access to Hugging Face for the initial model download

## Project Structure

```text
mac_dit/
├── image/                  # Generated images
├── model/                  # Model caches and quantized checkpoints
├── scripts/                # Setup, conversion, and run scripts
├── src/
│   ├── mac_dit/            # Reusable core package
│   │   ├── cli.py          # PyTorch command-line interface
│   │   ├── config.py       # Shared configuration and defaults
│   │   ├── formatting.py   # Memory and parameter formatting
│   │   ├── hardware.py     # Mac GPU and MPS information
│   │   ├── model_info.py   # DiT model statistics
│   │   ├── pipeline.py     # PyTorch loading and generation
│   │   ├── benchmark.py    # Quantization benchmarks
│   │   ├── mlx_backend/    # MLX model, QMM operators, conversion, and pipeline
│   │   └── quantization/   # PyTorch quantization algorithms and backends
│   ├── check_backend.py    # MPS availability check
│   ├── benchmark_quantization.py # Quantization benchmark entry point
│   ├── convert_to_mlx.py   # Diffusers-to-MLX conversion entry point
│   ├── run_mlx_dit.py      # MLX INT4 generation entry point
│   ├── model_config_fetch.py # Model information entry point
│   ├── mps_config_fetch.py # Mac hardware information entry point
│   └── run_dit.py          # PyTorch generation entry point
├── .venv/                  # Local Python virtual environment
└── .gitignore
```

Only documentation files inside `model/` and `.venv/` are committed. Downloaded weights, caches, converted checkpoints, and virtual-environment files are ignored by Git.

## Quick Start

Run these commands from the project root.

### 1. Install uv

```bash
brew install uv
```

Or run:

```bash
bash scripts/step1_uv_install.sh
```

### 2. Create the Environment and Install Dependencies

```bash
uv venv
uv pip install torch diffusers transformers accelerate Pillow huggingface_hub safetensors mlx
```

Activate the environment if needed:

```bash
# zsh or bash
source .venv/bin/activate

# fish
source .venv/bin/activate.fish
```

`scripts/step2_env_setup.sh` uses the fish activation script and should be sourced from fish:

```fish
source scripts/step2_env_setup.sh
```

### 3. Sign In to Hugging Face

```bash
uv run hf auth login
```

Or run:

```bash
bash scripts/step3_hf_login.sh
```

### 4. Check MPS

```bash
uv run python src/check_backend.py
```

Or run:

```bash
bash scripts/step4_check_backend.sh
```

Display complete Mac, GPU, and MPS information:

```bash
uv run python src/mps_config_fetch.py
```

Display DiT parameter counts, dtypes, and estimated memory usage:

```bash
uv run python src/model_config_fetch.py
```

### 5. Generate an Image with PyTorch MPS

```bash
uv run python src/run_dit.py
```

For example, generate an ImageNet tabby cat with 25 inference steps:

```bash
uv run python src/run_dit.py --class-label 281 --steps 25 --seed 42
```

The model is downloaded from Hugging Face on the first run and cached under `model/`. Generated images use this naming convention:

```text
image/dit_generated_image_<class>_<precision>_seed<seed>.png
```

## PyTorch Weight Quantization

The PyTorch path provides INT8 and INT4 weight-only quantization. By default, only attention and feed-forward Linear layers in the DiT Transformer are quantized. AdaNorm, timestep embeddings, the final output projection, and the VAE remain FP16.

### INT8 W8A16

```bash
uv run python src/run_dit.py \
  --class-label 281 \
  --seed 42 \
  --quantization int8
```

INT8 uses symmetric per-output-channel weight quantization with FP16 activations.

### INT4 W4A16

```bash
uv run python src/run_dit.py \
  --class-label 281 \
  --seed 42 \
  --quantization int4 \
  --group-size 128
```

INT4 uses symmetric group-wise quantization and packs two INT4 values into one byte. Group sizes of `64` and `128` are useful starting points.

### Change the Quantization Scope

Quantize every Linear layer:

```bash
uv run python src/run_dit.py --quantization int8 --quantize-all-linears
```

Exclude sensitive modules by name:

```bash
uv run python src/run_dit.py \
  --quantization int8 \
  --exclude-layer transformer_blocks.0 \
  --exclude-layer transformer_blocks.27
```

### Save and Load PyTorch Quantized Weights

```bash
# Quantize and save without generating an image. Default destination:
# model/quantized/DiT-XL-2-256-int4-g128/
uv run python src/run_dit.py \
  --quantization int4 \
  --group-size 128 \
  --quantize-only

# Or specify a destination explicitly
uv run python src/run_dit.py \
  --quantization int4 \
  --quantize-only \
  --save-quantized ./model/quantized/my-dit-int4

# Load a saved quantized Transformer
uv run python src/run_dit.py \
  --load-quantized ./model/quantized/DiT-XL-2-256-int4-g128

# Load the same weights with another execution backend
uv run python src/run_dit.py \
  --load-quantized ./model/quantized/DiT-XL-2-256-int4-g128 \
  --quant-backend metal
```

A quantized directory contains `transformer.safetensors` and `quantization.json`. Store these directories under `model/` so large weights remain excluded from Git.

### PyTorch Quantization Benchmarks

Use the same class, seed, scheduler, and number of steps for FP16, INT8, and INT4:

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

The `reference` backend dequantizes weights before each Linear call and then invokes floating-point `F.linear`. It is useful for correctness and memory-compression checks, but is not expected to outperform FP16.

An experimental custom Metal backend is also available:

```bash
uv run python src/run_dit.py \
  --quantization int4 --quant-backend metal
```

This backend uses `torch.mps.compile_shader` for custom W8A16/W4A16 Linear kernels. It reads INT8 or packed INT4 weights directly, keeps activations in FP16, and accumulates in FP32. The implementation favors readability and correctness; it does not yet include matrix tiling, SIMD cooperation, or threadgroup caching. Use the MLX backend for practical low-bit acceleration.

## MLX Accelerated Backend

MLX is the recommended low-bit execution path. The complete 28-layer DiT Transformer remains in MLX/Metal instead of converting tensors between PyTorch and MLX for every Linear layer. Only the small latent is passed to the CPU scheduler after each diffusion step, and the VAE runs once at the end.

### 1. Convert and Quantize the Weights

```bash
uv run python src/convert_to_mlx.py --bits 4 --group-size 128
```

The default output is:

```text
model/mlx/DiT-XL-2-256-w4-g128/
├── dit.safetensors  # MLX FP16 and packed INT4 parameters
└── mlx_config.json  # Architecture, quantization settings, and layer list
```

By default, 168 attention and FFN Linear layers are quantized. Use `--bits 8` for 8-bit weights or `--output-dir` to select another directory. Conversion only needs to be performed once.

### 2. Generate an Image with MLX

```bash
uv run python src/run_mlx_dit.py \
  --class-label 281 \
  --steps 25 \
  --seed 42
```

`mx.compile` is enabled by default. The first run compiles the computation graph, while later runs are faster. Add `--no-compile` when debugging.

On the M3 MacBook Air used for this project, with a 10-core GPU and 16 GB of unified memory, class 281, seed 42, and 25 inference steps produced these same-scope measurements:

- PyTorch MPS FP16: approximately 6.02 seconds total.
- MLX INT4 group-128: approximately 4.96 seconds for DiT denoising and 5.41 seconds total, with approximately 0.99 GB peak MLX memory.
- MLX with only the FFN quantized: approximately 5.55 seconds total, so attention and FFN quantization remains the default.

Results vary with first-run compilation, temperature, and system load. Use warmups and compare the median of multiple runs for final performance decisions.

### MLX Code Interfaces

The MLX implementation separates model structure, operators, conversion, and runtime concerns:

- `mlx_backend/operators.py`: calls `mx.quantized_matmul` and fused scaled dot-product attention.
- `mlx_backend/model.py`: implements PatchEmbed, AdaLayerNormZero, Attention, FFN, and the complete DiT Transformer.
- `mlx_backend/conversion.py`: handles PyTorch OIHW to MLX OHWI conversion, parameter renaming, and checkpoint serialization.
- `mlx_backend/pipeline.py`: implements classifier-free guidance, scheduler bridging, VAE decoding, and timing.
- `mlx_backend/cli.py`: provides conversion and inference command-line interfaces.

The main quantized operator interface is:

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

The full model interface is:

```python
noise_and_sigma = model(
    sample,        # [B, 4, 32, 32], NCHW FP16
    timesteps,     # [B]
    class_labels,  # [B], with 1000 as the null class
)
```

Load an existing MLX checkpoint:

```python
from mac_dit.mlx_backend.conversion import load_mlx_transformer

model, manifest = load_mlx_transformer(
    "model/mlx/DiT-XL-2-256-w4-g128"
)
```

## Generation Options

Run `python src/run_dit.py --help` or `python src/run_mlx_dit.py --help` for every available option. Common arguments include:

- `--class-label`: ImageNet class index; `281` is a tabby cat.
- `--steps`: Number of inference steps.
- `--seed`: Random seed for reproducible comparisons.
- `--output-dir`: Generated-image directory.
- `--model-id`: Hugging Face model ID.
- `--cache-dir`: Downloaded-model cache directory.

Reuse the PyTorch path from Python:

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

## Troubleshooting

### MPS or Metal Is Unavailable

Run the backend check and verify that the Mac, macOS, PyTorch, and terminal environment can access Metal. MLX should be run from a normal macOS terminal with GPU access; a sandboxed or headless process may not expose the Metal device.

### Model Download Is Interrupted

Run the command again. Hugging Face will reuse files already present in `model/` and continue the download.

### Recreate the Virtual Environment

Remove the local `.venv/` directory, then repeat the environment installation commands.
