from warnings import warn
import numpy as np


GPU_AVAIL = True
try:
    import cupy as cp
except:
    GPU_AVAIL = False
    warn("GPU (cupy) not available. Falling back to CPU (numpy) computations.")


def get_d__(device):
    if device == "cpu":
        return np, device
    elif isinstance(device, str) and device.startswith("cuda"):
        if not GPU_AVAIL:
            raise RuntimeError("GPU (cupy) not available.")
        if device != "cuda":
            try:
                _, index = device.split(":", 1)
                cp.cuda.Device(int(index)).use()
            except ValueError as exc:
                raise ValueError("Invalid CUDA device format, expected 'cuda:<index>'.") from exc
        return cp, device
    else:
        raise ValueError("Unknown value passed as device")