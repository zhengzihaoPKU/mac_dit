from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class MlxDiTConfig:
    """MLX DiT 网络结构配置，与 Diffusers Transformer2DModel 对齐。"""

    sample_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    out_channels: int = 8
    num_layers: int = 28
    num_attention_heads: int = 16
    attention_head_dim: int = 72
    num_classes: int = 1000
    norm_eps: float = 1e-5
    ada_norm_eps: float = 1e-6

    @property
    def hidden_size(self):
        return self.num_attention_heads * self.attention_head_dim

    @property
    def patch_tokens(self):
        return (self.sample_size // self.patch_size) ** 2

    @classmethod
    def from_diffusers_config(cls, config):
        read = lambda name, default=None: (
            config.get(name, default)
            if isinstance(config, dict)
            else getattr(config, name, default)
        )
        return cls(
            sample_size=read("sample_size"),
            patch_size=read("patch_size"),
            in_channels=read("in_channels"),
            out_channels=read("out_channels"),
            num_layers=read("num_layers"),
            num_attention_heads=read("num_attention_heads"),
            attention_head_dim=read("attention_head_dim"),
            num_classes=read("num_embeds_ada_norm"),
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MlxQuantizationConfig:
    """MLX 原生权重量化或权重-激活量化配置。"""

    bits: int = 4
    group_size: int = 128
    mode: str = "affine"
    quantize_activations: bool = False
    include_patterns: tuple[str, ...] = (".attn1.", ".ff.")
    exclude_patterns: tuple[str, ...] = ()

    def __post_init__(self):
        if self.quantize_activations:
            specifications = {
                "mxfp8": (8, 32),
                "nvfp4": (4, 16),
            }
            if self.mode not in specifications:
                raise ValueError("激活量化仅支持 mxfp8 或 nvfp4")
            expected = specifications[self.mode]
            if (self.bits, self.group_size) != expected:
                raise ValueError(
                    f"{self.mode} 必须使用 bits={expected[0]}、"
                    f"group_size={expected[1]}"
                )
        else:
            if self.bits not in (4, 8):
                raise ValueError("MLX DiT 当前只支持 4-bit 或 8-bit 权重量化")
            if self.group_size <= 0 or self.group_size % 32 != 0:
                raise ValueError("MLX group_size 必须是 32 的正整数倍")
            if self.mode != "affine":
                raise ValueError("weight-only 量化当前只支持 affine 模式")

    @property
    def label(self):
        if self.quantize_activations:
            return f"mlx-{self.mode}-w{self.bits}a{self.bits}-g{self.group_size}"
        return f"mlx-w{self.bits}a16-g{self.group_size}"

    def matches(self, path):
        included = not self.include_patterns or any(
            pattern in path for pattern in self.include_patterns
        )
        excluded = any(pattern in path for pattern in self.exclude_patterns)
        return included and not excluded

    def to_dict(self):
        data = asdict(self)
        data["include_patterns"] = list(self.include_patterns)
        data["exclude_patterns"] = list(self.exclude_patterns)
        return data

    @classmethod
    def from_dict(cls, data):
        return cls(
            bits=data["bits"],
            group_size=data["group_size"],
            mode=data.get("mode", "affine"),
            quantize_activations=data.get("quantize_activations", False),
            include_patterns=tuple(data.get("include_patterns", ())),
            exclude_patterns=tuple(data.get("exclude_patterns", ())),
        )
