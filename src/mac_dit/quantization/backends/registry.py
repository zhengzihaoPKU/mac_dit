from .metal import MetalBackend
from .reference import ReferenceBackend


_BACKENDS = {}


def register_backend(backend):
    if not getattr(backend, "name", None):
        raise ValueError("量化后端必须定义 name")
    _BACKENDS[backend.name] = backend


def get_backend(name):
    try:
        return _BACKENDS[name]
    except KeyError as error:
        choices = ", ".join(sorted(_BACKENDS)) or "无"
        raise ValueError(f"未知量化后端 {name}，可用后端: {choices}") from error


def available_backends():
    return tuple(sorted(_BACKENDS))


register_backend(ReferenceBackend())
register_backend(MetalBackend())
