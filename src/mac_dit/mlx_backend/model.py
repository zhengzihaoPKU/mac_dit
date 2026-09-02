"""使用 MLX 实现的 class-conditional DiT Transformer。

公开接口保持 NCHW，方便与 Diffusers scheduler/VAE 对接；卷积内部转换为 MLX 的 NHWC。
"""

import math

import mlx.core as mx
import mlx.nn as nn

from .operators import scaled_dot_product_attention


def timestep_embedding(timesteps, embedding_dim=256, max_period=10000):
    """与 Diffusers flip_sin_to_cos=True、freq_shift=1 的实现对齐。"""
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - 1)
    frequencies = mx.exp(exponent)
    arguments = timesteps.astype(mx.float32)[:, None] * frequencies[None]
    embedding = mx.concatenate([mx.cos(arguments), mx.sin(arguments)], axis=-1)
    if embedding_dim % 2:
        embedding = mx.pad(embedding, [(0, 0), (0, 1)])
    return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_1 = nn.Linear(256, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def __call__(self, timesteps, dtype):
        hidden = timestep_embedding(timesteps).astype(dtype)
        hidden = nn.silu(self.linear_1(hidden))
        return self.linear_2(hidden)


class LabelEmbedding(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        # 额外一行是 classifier-free guidance 使用的空类别。
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def __call__(self, class_labels):
        return self.embedding_table(class_labels)


class CombinedTimestepLabelEmbeddings(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(hidden_size)
        self.class_embedder = LabelEmbedding(num_classes, hidden_size)

    def __call__(self, timesteps, class_labels, dtype):
        return self.timestep_embedder(timesteps, dtype) + self.class_embedder(
            class_labels
        )


class AdaLayerNormZero(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.emb = CombinedTimestepLabelEmbeddings(
            config.num_classes,
            hidden_size,
        )
        self.linear = nn.Linear(hidden_size, hidden_size * 6)
        self.norm = nn.LayerNorm(
            hidden_size,
            eps=config.ada_norm_eps,
            affine=False,
        )

    def __call__(self, hidden_states, timesteps, class_labels):
        conditioning = self.emb(timesteps, class_labels, hidden_states.dtype)
        modulation = self.linear(nn.silu(conditioning))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(
            modulation,
            6,
            axis=-1,
        )
        normalized = self.norm(hidden_states)
        normalized = normalized * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(hidden_size, hidden_size)
        self.to_v = nn.Linear(hidden_size, hidden_size)
        self.to_out = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states):
        batch, tokens, _ = hidden_states.shape

        def split_heads(projection):
            projection = projection.reshape(
                batch,
                tokens,
                self.num_heads,
                self.head_dim,
            )
            return mx.transpose(projection, (0, 2, 1, 3))

        query = split_heads(self.to_q(hidden_states))
        key = split_heads(self.to_k(hidden_states))
        value = split_heads(self.to_v(hidden_states))
        attended = scaled_dot_product_attention(
            query,
            key,
            value,
            head_dim=self.head_dim,
        )
        attended = mx.transpose(attended, (0, 2, 1, 3)).reshape(
            batch,
            tokens,
            self.num_heads * self.head_dim,
        )
        return self.to_out(attended)


class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size * 4)
        self.out = nn.Linear(hidden_size * 4, hidden_size)

    def __call__(self, hidden_states):
        return self.out(nn.gelu_approx(self.proj(hidden_states)))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = AdaLayerNormZero(config)
        self.attn1 = Attention(config)
        self.norm3 = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            affine=False,
        )
        self.ff = FeedForward(config.hidden_size)

    def __call__(self, hidden_states, timesteps, class_labels):
        normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states,
            timesteps,
            class_labels,
        )
        hidden_states = hidden_states + gate_msa[:, None] * self.attn1(normalized)

        normalized = self.norm3(hidden_states)
        normalized = normalized * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = hidden_states + gate_mlp[:, None] * self.ff(normalized)
        return hidden_states


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position = mx.zeros(
            (1, config.patch_tokens, config.hidden_size),
            dtype=mx.float32,
        )

    def __call__(self, sample):
        # 对外 NCHW；MLX Conv2d 使用 NHWC。
        sample = mx.transpose(sample, (0, 2, 3, 1))
        patches = self.proj(sample)
        patches = patches.reshape(patches.shape[0], -1, patches.shape[-1])
        return (patches + self.position).astype(patches.dtype)


class MlxDiT(nn.Module):
    """完整 MLX DiT Transformer，调用格式与 Diffusers Transformer 接近。"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_embed = PatchEmbed(config)
        self.transformer_blocks = [
            TransformerBlock(config) for _ in range(config.num_layers)
        ]
        self.norm_out = nn.LayerNorm(
            config.hidden_size,
            eps=config.ada_norm_eps,
            affine=False,
        )
        self.proj_out_1 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        patch_output = config.patch_size * config.patch_size * config.out_channels
        self.proj_out_2 = nn.Linear(config.hidden_size, patch_output)

    def __call__(self, sample, timesteps, class_labels):
        """执行 DiT 前向。

        Args:
            sample: NCHW latent，形状 [B, 4, 32, 32]。
            timesteps: 形状 [B]。
            class_labels: 形状 [B]，空类别编号为 1000。
        Returns:
            NCHW noise/sigma，形状 [B, 8, 32, 32]。
        """
        hidden_states = self.pos_embed(sample)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, timesteps, class_labels)

        conditioning = self.transformer_blocks[0].norm1.emb(
            timesteps,
            class_labels,
            hidden_states.dtype,
        )
        shift, scale = mx.split(
            self.proj_out_1(nn.silu(conditioning)),
            2,
            axis=-1,
        )
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states)

        patch = self.config.patch_size
        side = self.config.sample_size // patch
        hidden_states = hidden_states.reshape(
            -1,
            side,
            side,
            patch,
            patch,
            self.config.out_channels,
        )
        hidden_states = mx.transpose(hidden_states, (0, 5, 1, 3, 2, 4))
        return hidden_states.reshape(
            -1,
            self.config.out_channels,
            self.config.sample_size,
            self.config.sample_size,
        )
