from engine.Tensor import Tensor
from nn.act_func import softmax

def _check_if_same_device(y1: Tensor, y2: Tensor):
    if y1.device != y2.device:
        raise RuntimeError(
            "Expected all tensors to be on the same device, but found at least two devices, cuda and cpu!"
        )

def cross_entropy_loss(input: Tensor, target: Tensor, reduction='mean') -> Tensor:
    """
    参数:
        input (Tensor): (batch_size, num_classes).
        target (Tensor): (batch_size,).
        reduction (str, optional): The reduction method. Can be 'none', 'mean', or 'sum'. Default: 'mean'.

    返回值:
        Tensor: The computed cross-entropy loss.
    """
    # Ensure input and target are on the same device
    _check_if_same_device(input, target)

    # Compute softmax probabilities
    softmax_output = softmax(input, dim=-1)

    # Compute cross-entropy loss
    loss = -softmax_output[range(len(softmax_output)), target.data.astype(int)].log()

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

