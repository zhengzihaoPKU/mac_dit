from functools import lru_cache
from pathlib import Path

import torch


KERNEL_PATH = Path(__file__).with_name("kernels") / "dequantize.metal"


@lru_cache(maxsize=32)
def _mps_uint(value: int) -> torch.Tensor:
    """为 Metal 常量参数创建可复用的标量缓冲区。"""
    return torch.tensor([value], dtype=torch.int32, device="mps")


@lru_cache(maxsize=1)
def _shader_library():
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("当前 PyTorch 不支持 torch.mps.compile_shader")
    return torch.mps.compile_shader(KERNEL_PATH.read_text(encoding="utf-8"))


def _dispatch(kernel, *args, numel):
    group_size = min(256, numel)
    kernel(
        *args,
        threads=[numel, 1, 1],
        group_size=[group_size, 1, 1],
    )


class MetalBackend:
    """直接使用打包低比特权重执行 W8A16/W4A16 Linear。"""

    name = "metal"

    def supports(self, weight, device):
        return (
            device.type == "mps"
            and torch.backends.mps.is_available()
            and hasattr(torch.mps, "compile_shader")
            and weight.config.mode in ("int8", "int4")
        )

    def _linear(self, inputs, weight, bias):
        library = _shader_library()
        out_features, in_features = weight.original_shape
        if inputs.shape[-1] != in_features:
            raise ValueError(
                f"输入最后一维应为 {in_features}，实际为 {inputs.shape[-1]}"
            )

        flat_inputs = inputs.contiguous().reshape(-1, in_features)
        rows = flat_inputs.shape[0]
        output = torch.empty(
            (rows, out_features),
            device=inputs.device,
            dtype=torch.float16,
        )
        bias_buffer = (
            bias.to(dtype=torch.float16).contiguous()
            if bias is not None
            else torch.empty(1, device=inputs.device, dtype=torch.float16)
        )
        numel = output.numel()
        if numel == 0:
            return output.reshape(*inputs.shape[:-1], out_features)

        if weight.config.mode == "int8":
            _dispatch(
                library.linear_int8,
                flat_inputs,
                weight.qweight,
                weight.scales,
                bias_buffer,
                output,
                _mps_uint(rows),
                _mps_uint(in_features),
                _mps_uint(out_features),
                _mps_uint(int(bias is not None)),
                _mps_uint(numel),
                numel=numel,
            )
        else:
            _dispatch(
                library.linear_int4,
                flat_inputs,
                weight.qweight,
                weight.scales,
                bias_buffer,
                output,
                _mps_uint(rows),
                _mps_uint(in_features),
                _mps_uint(weight.padded_in_features),
                _mps_uint(out_features),
                _mps_uint(weight.config.group_size),
                _mps_uint(int(bias is not None)),
                _mps_uint(numel),
                numel=numel,
            )

        return output.reshape(*inputs.shape[:-1], out_features)

    def linear(self, inputs, weight, bias):
        if not self.supports(weight, inputs.device):
            raise RuntimeError(
                "Metal 量化后端需要支持 compile_shader 的 PyTorch MPS 环境"
            )
        if inputs.dtype != torch.float16:
            raise TypeError("Metal 量化后端当前只支持 FP16 激活")
        return self._linear(inputs, weight, bias)
