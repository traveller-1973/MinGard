
from warnings import warn
import numpy as np


GPU_AVAIL = True
try:
    import cupy as cp
except:
    GPU_AVAIL = False
    warn("GPU (cupy) not available. Falling back to CPU (numpy) computations.")


def get_device(device):
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

def dLex():
    return cp if GPU_AVAIL else np


def broadcast_axis__(shape_left, shape_right):
    # 输入为两个Tensor的形状
    # 返回每个tensor需要广播的维度
    if shape_left == shape_right:
        return ((), ())

    left_dim = len(shape_left)
    right_dim = len(shape_right)
    result_ndim = max(left_dim, right_dim)

    # 在高位填充1维度到最大维度
    left_padded = (1,) * (result_ndim - left_dim) + shape_left
    right_padded = (1,) * (result_ndim - right_dim) + shape_right

    left_axes = []
    right_axes = []

    # 比较每个维度，较小的维度需要广播
    for axis_idx, (left_axis, right_axis) in enumerate(zip(left_padded, right_padded)):
        if right_axis > left_axis:
            left_axes.append(axis_idx)
        elif left_axis > right_axis:
            right_axes.append(axis_idx)

    return tuple(left_axes), tuple(right_axes)