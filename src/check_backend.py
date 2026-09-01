import torch

# 检查 MPS (Apple Silicon GPU) 是否可用
def check_backend():
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
            print("Please install the PyTorch nightlies to use MPS acceleration.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")
        # 如果沒有 MPS，就直接退出
        exit()

if __name__ == "__main__":
    check_backend()