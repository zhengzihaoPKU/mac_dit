from mac_dit.hardware import MPSUnavailableError, require_mps


def check_backend():
    """兼容原有调用方式，检查当前环境是否支持 MPS。"""
    try:
        print(require_mps())
    except MPSUnavailableError as error:
        print(error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    check_backend()
