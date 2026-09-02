"""Apple MLX DiT 后端。

本模块不在包初始化时导入 mlx，避免普通 PyTorch 命令依赖 Metal 环境。
"""

from .config import MlxDiTConfig, MlxQuantizationConfig

__all__ = ["MlxDiTConfig", "MlxQuantizationConfig"]
