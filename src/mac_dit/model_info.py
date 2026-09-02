import argparse

from .config import DEFAULT_CACHE_DIR, DEFAULT_MODEL_ID, GenerationConfig
from .formatting import bytes_to_gb, format_parameter_count


def _get_config_value(config, key, default="未知"):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def fetch_model_config(pipe, model_id=None):
    """获取 DiT pipeline 中 Transformer 模型的配置和内存信息。"""
    model = getattr(pipe, "transformer", pipe)
    model_config = getattr(model, "config", None)
    pipeline_config = getattr(pipe, "config", None)

    parameters = list(model.parameters())
    buffers = list(model.buffers())
    total_parameters = sum(parameter.numel() for parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for parameter in parameters if parameter.requires_grad
    )
    model_size_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in parameters + buffers
    )
    dtypes = sorted({str(parameter.dtype).removeprefix("torch.") for parameter in parameters})

    resolved_model_id = model_id
    if not resolved_model_id:
        resolved_model_id = _get_config_value(pipeline_config, "_name_or_path", None)
    if not resolved_model_id:
        resolved_model_id = _get_config_value(model_config, "_name_or_path", None)
    if not resolved_model_id:
        resolved_model_id = model.__class__.__name__

    return {
        "model_id": resolved_model_id,
        "architecture": model.__class__.__name__,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "dtypes": dtypes or ["未知"],
        "estimated_size_gb": bytes_to_gb(model_size_bytes),
        "sample_size": _get_config_value(model_config, "sample_size"),
        "input_channels": _get_config_value(model_config, "in_channels"),
        "layers": _get_config_value(model_config, "num_layers"),
        "attention_heads": _get_config_value(model_config, "num_attention_heads"),
        "classes": _get_config_value(model_config, "num_embeds_ada_norm"),
    }


def print_model_config(pipe, model_id=None):
    """以易读格式输出 DiT 模型配置。"""
    config = fetch_model_config(pipe, model_id)

    print("=== DiT 模型信息 ===")
    print(f"模型名称: {config['model_id']}")
    print(f"模型架构: {config['architecture']}")
    print(
        "参数量: "
        f"{format_parameter_count(config['total_parameters'])} "
        f"({config['total_parameters']:,})"
    )
    print(
        "可训练参数量: "
        f"{format_parameter_count(config['trainable_parameters'])} "
        f"({config['trainable_parameters']:,})"
    )
    print(f"数据类型: {', '.join(config['dtypes'])}")
    print(f"参数与缓冲区估算内存: {config['estimated_size_gb']:.2f} GB")
    print(f"输入图像尺寸: {config['sample_size']}")
    print(f"输入通道数: {config['input_channels']}")
    print(f"Transformer 层数: {config['layers']}")
    print(f"注意力头数: {config['attention_heads']}")
    print(f"类别数: {config['classes']}")


def build_parser():
    parser = argparse.ArgumentParser(description="输出 Hugging Face DiT 模型配置")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="模型 ID")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="模型缓存目录",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from .pipeline import load_pipeline

    config = GenerationConfig(model_id=args.model_id, cache_dir=args.cache_dir)
    pipe = load_pipeline(config, move_to_device=False)
    print_model_config(pipe, config.model_id)
