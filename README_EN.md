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
│   ├── main.py               # Unified command-line entry point
│   └── mac_dit/               # Reusable core package
│       ├── app.py             # Subcommand dispatch and lazy imports
│       ├── cli.py             # PyTorch generation arguments
│       ├── pipeline.py        # PyTorch loading and inference
│       ├── hardware.py        # Mac GPU and MPS information
│       ├── mlx_backend/       # MLX model, operators, conversion, and inference
│       └── quantization/      # PyTorch quantization algorithms and operators
├── .venv/                  # Local Python virtual environment
└── .gitignore
```

Only documentation files inside `model/` and `.venv/` are committed. Downloaded weights, caches, converted checkpoints, and virtual-environment files are ignored by Git.

Every operation is available from one entry point:

```bash
uv run python src/main.py --help
```

## Quick Start

Run these commands from the project root.

### 1. Install uv

```bash
brew install uv
```

Or install uv, create the virtual environment, and install every dependency with one script:

```bash
bash scripts/setup.sh
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

### 3. Sign In to Hugging Face

```bash
uv run hf auth login
```

### 4. Check MPS

```bash
uv run python src/main.py check
```

Display complete Mac, GPU, and MPS information:

```bash
uv run python src/main.py hardware
```

Display DiT parameter counts, dtypes, and estimated memory usage:

```bash
uv run python src/main.py model-info
```

### 5. Generate an Image with PyTorch MPS

```bash
uv run python src/main.py generate
```

For example, generate an ImageNet tabby cat with 25 inference steps:

```bash
uv run python src/main.py generate --class-label 281 --steps 25 --seed 42
```

The model is downloaded from Hugging Face on the first run and cached under `model/`. Generated images use this naming convention:

```text
image/dit_generated_image_<class>_<precision>_seed<seed>.png
```

## PyTorch Weight Quantization

The PyTorch path provides INT8 and INT4 weight-only quantization. By default, only attention and feed-forward Linear layers in the DiT Transformer are quantized. AdaNorm, timestep embeddings, the final output projection, and the VAE remain FP16.

### INT8 W8A16

```bash
uv run python src/main.py generate \
  --class-label 281 \
  --seed 42 \
  --quantization int8
```

INT8 uses symmetric per-output-channel weight quantization with FP16 activations.

### INT4 W4A16

```bash
uv run python src/main.py generate \
  --class-label 281 \
  --seed 42 \
  --quantization int4 \
  --group-size 128
```

INT4 uses symmetric group-wise quantization and packs two INT4 values into one byte. Group sizes of `64` and `128` are useful starting points.

### Change the Quantization Scope

Quantize every Linear layer:

```bash
uv run python src/main.py generate --quantization int8 --quantize-all-linears
```

Exclude sensitive modules by name:

```bash
uv run python src/main.py generate \
  --quantization int8 \
  --exclude-layer transformer_blocks.0 \
  --exclude-layer transformer_blocks.27
```

### Save and Load PyTorch Quantized Weights

```bash
# Quantize and save without generating an image. Default destination:
# model/quantized/DiT-XL-2-256-int4-g128/
uv run python src/main.py generate \
  --quantization int4 \
  --group-size 128 \
  --quantize-only

# Or specify a destination explicitly
uv run python src/main.py generate \
  --quantization int4 \
  --quantize-only \
  --save-quantized ./model/quantized/my-dit-int4

# Load a saved quantized Transformer
uv run python src/main.py generate \
  --load-quantized ./model/quantized/DiT-XL-2-256-int4-g128

# Load the same weights with another execution backend
uv run python src/main.py generate \
  --load-quantized ./model/quantized/DiT-XL-2-256-int4-g128 \
  --quant-backend metal
```

A quantized directory contains `transformer.safetensors` and `quantization.json`. Store these directories under `model/` so large weights remain excluded from Git.

### PyTorch Quantization Benchmarks

Use the same class, seed, scheduler, and number of steps for FP16, INT8, and INT4:

```bash
# FP16
uv run python src/main.py benchmark \
  --class-label 281 --seed 42 --steps 25 --warmup 1 --repeats 3

# INT8
uv run python src/main.py benchmark \
  --class-label 281 --seed 42 --steps 25 \
  --quantization int8 --warmup 1 --repeats 3 \
  --json ./benchmark-int8.json

# INT4
uv run python src/main.py benchmark \
  --class-label 281 --seed 42 --steps 25 \
  --quantization int4 --group-size 128 --warmup 1 --repeats 3
```

The `reference` backend dequantizes weights before each Linear call and then invokes floating-point `F.linear`. It is useful for correctness and memory-compression checks, but is not expected to outperform FP16.

An experimental custom Metal backend is also available:

```bash
uv run python src/main.py generate \
  --quantization int4 --quant-backend metal
```

This backend uses `torch.mps.compile_shader` for custom W8A16/W4A16 Linear kernels. It reads INT8 or packed INT4 weights directly, keeps activations in FP16, and accumulates in FP32. The implementation favors readability and correctness; it does not yet include matrix tiling, SIMD cooperation, or threadgroup caching. Use the MLX backend for practical low-bit acceleration.

## MLX Accelerated Backend

MLX is the recommended low-bit execution path. The complete 28-layer DiT Transformer remains in MLX/Metal instead of converting tensors between PyTorch and MLX for every Linear layer. Only the small latent is passed to the CPU scheduler after each diffusion step, and the VAE runs once at the end.

### 1. Convert and Quantize the Weights

```bash
uv run python src/main.py mlx-convert --bits 4 --group-size 128
```

The default output is:

```text
model/mlx/DiT-XL-2-256-w4-g128/
├── dit.safetensors  # MLX FP16 and packed INT4 parameters
└── mlx_config.json  # Architecture, quantization settings, and layer list
```

By default, 168 attention and FFN Linear layers are quantized. Use `--bits 8` for 8-bit weights or `--output-dir` to select another directory. Conversion only needs to be performed once.

### Quantize Both Weights and Activations

The project supports MLX's native `mx.qqmm`. Weights are stored in low precision in the checkpoint, while activations are dynamically quantized for each Linear call before the fused low-precision Metal matrix multiplication. DiT biases are added in floating point after `qqmm`.

MXFP8 W8A8 with its required group size of 32:

```bash
uv run python src/main.py mlx-convert --activation-quantization mxfp8
uv run python src/main.py mlx-generate \
  --mlx-model-dir model/mlx/DiT-XL-2-256-mxfp8-w8a8 \
  --class-label 281 --steps 25 --seed 42
```

NVFP4 W4A4 with its required group size of 16:

```bash
uv run python src/main.py mlx-convert --activation-quantization nvfp4
uv run python src/main.py mlx-generate \
  --mlx-model-dir model/mlx/DiT-XL-2-256-nvfp4-w4a4 \
  --class-label 281 --steps 25 --seed 42
```

Both modes quantize 168 attention and FFN Linear layers by default. AdaNorm, embeddings, the final projection, and the VAE remain floating point. `--quantize-all-linears` and `--exclude-layer` can still change the scope.

### 2. Generate an Image with MLX

```bash
uv run python src/main.py mlx-generate \
  --class-label 281 \
  --steps 25 \
  --seed 42
```

`mx.compile` is enabled by default. The first run compiles the computation graph, while later runs are faster. Add `--no-compile` when debugging.

### Measured Speed Comparison

These measurements use an M3 MacBook Air with a 10-core GPU and 16 GB of unified memory. Every mode uses class 281, seed 42, and 25 inference steps. Total latency includes DiT denoising and the final VAE decode, but excludes model loading and checkpoint conversion. Higher relative FP16 throughput is better.

| Rank | Execution path | Weight / activation | Total latency | Relative FP16 throughput |
| ---: | --- | --- | ---: | ---: |
| 1 | MLX QMM, attention + FFN (default) | INT4 / FP16 | **5.45 s** | **1.10×** |
| 2 | MLX QMM, FFN only | INT4 / FP16 | 5.55 s | 1.08× |
| 3 | PyTorch MPS baseline | FP16 / FP16 | 6.02 s | 1.00× |
| 4 | PyTorch reference | INT8 / FP16 | 8.59 s | 0.70× |
| 5 | PyTorch reference | INT4 / FP16 | 11.23 s | 0.54× |
| 6 | MLX QQMM NVFP4 | FP4 / FP4 | 17.49 s | 0.34× |
| 7 | MLX QQMM MXFP8 | FP8 / FP8 | 18.52 s | 0.33× |
| 8 | PyTorch custom Metal baseline kernel | INT4 / FP16 | 232.62 s | 0.03× |

Conclusion: MLX W4A16 is the fastest path on this M3, approximately 10% faster than the PyTorch FP16 baseline, with about 0.99 GB peak MLX memory. W8A8 and W4A4 through `mx.qqmm` dynamically quantize activations and do not provide a speed benefit on M3. The custom Metal kernel demonstrates direct packed-INT4 access but lacks matrix tiling and SIMD optimization, so it is not suitable for practical inference. Compilation, temperature, and system load affect individual runs; use warmups and compare medians for formal benchmarking.

### MLX Code Interfaces

The MLX implementation separates model structure, operators, conversion, and runtime concerns:

- `mlx_backend/operators.py`: calls `mx.quantized_matmul`, `mx.qqmm`, and fused scaled dot-product attention.
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

The weight-and-activation quantized operator interface is:

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

Run `python src/main.py generate --help` or `python src/main.py mlx-generate --help` for every available option. Common arguments include:

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
