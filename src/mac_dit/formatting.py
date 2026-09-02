BYTES_PER_GB = 1024 ** 3


def bytes_to_gb(size_in_bytes):
    """将字节转换为 GiB，并使用用户更熟悉的 GB 标签展示。"""
    return size_in_bytes / BYTES_PER_GB


def format_parameter_count(count):
    """将参数量格式化为 K、M 或 B。"""
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f} B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f} M"
    if count >= 1_000:
        return f"{count / 1_000:.2f} K"
    return str(count)
