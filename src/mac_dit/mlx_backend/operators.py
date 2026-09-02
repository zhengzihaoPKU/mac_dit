"""MLX 高性能算子入口。

这里是量化计算真正发生的位置。模型代码只组合这些算子，不包含权重打包细节。
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path


def quantized_matmul(
    inputs,
    packed_weight,
    scales,
    quantization_biases=None,
    *,
    bits,
    group_size,
    mode="affine",
):
    """调用 MLX 原生 Metal QMM，不创建完整 FP16 权重矩阵。"""
    return mx.quantized_matmul(
        inputs,
        packed_weight,
        scales=scales,
        biases=quantization_biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def quantized_quantized_matmul(
    inputs,
    packed_weight,
    scales,
    *,
    bits,
    group_size,
    mode,
):
    """调用 MLX 原生 QQMM，在算子内部动态量化激活和计算。"""
    return mx.qqmm(
        inputs,
        packed_weight,
        scales=scales,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


class QuantizedLinear(nn.QuantizedLinear):
    """显式使用本项目 quantized_matmul 接口的 MLX Linear。"""

    def __call__(self, inputs):
        outputs = quantized_matmul(
            inputs,
            self.weight,
            self.scales,
            self.get("biases"),
            bits=self.bits,
            group_size=self.group_size,
            mode=self.mode,
        )
        if "bias" in self:
            outputs = outputs + self.bias
        return outputs


class QuantizedActivationLinear(nn.QQLinear):
    """支持 bias 的 W8A8/W4A4 Linear；MLX QQMM 本身不处理 bias。"""

    def __call__(self, inputs):
        outputs = quantized_quantized_matmul(
            inputs,
            self.weight,
            self.get("scales"),
            bits=self.bits,
            group_size=self.group_size,
            mode=self.mode,
        )
        if "bias" in self:
            outputs = outputs + self.bias
        return outputs

    @classmethod
    def from_linear(cls, linear, *, group_size, bits, mode):
        output_dims, input_dims = linear.weight.shape
        module = cls(input_dims, output_dims, group_size, bits, mode=mode)
        module.weight = linear.weight
        if linear.get("bias") is not None:
            module.bias = linear.bias
        module.train(linear.training)
        return module


def quantize_linear_modules(model, config):
    """原地将匹配的 nn.Linear 替换为 MLX 原生量化 Linear。"""
    quantized_paths = []

    def replace(path, module):
        if isinstance(module, nn.Linear) and config.matches(path):
            quantized_paths.append(path)
            linear_type = (
                QuantizedActivationLinear
                if config.quantize_activations
                else QuantizedLinear
            )
            return linear_type.from_linear(
                module,
                group_size=config.group_size,
                bits=config.bits,
                mode=config.mode,
            )
        return module

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(replace, leaves, is_leaf=nn.Module.is_module)
    model.update_modules(leaves)
    return tuple(quantized_paths)


def scaled_dot_product_attention(query, key, value, *, head_dim):
    """调用 MLX fused Metal attention，输入布局为 B,H,T,D。"""
    return mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=1.0 / math.sqrt(head_dim),
    )
