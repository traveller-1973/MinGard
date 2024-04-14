from engine.Tensor import Tensor

def relu(val: Tensor) -> Tensor:

    out = Tensor(val.d.maximum(0, val.data), val.device, val.dtype, (val,))

    if val.requires_grad and Tensor.grad_enabled:

        def backward():
            val.grad += (val.data > 0) * out.grad

        out._backward = backward
        out.requires_grad_(True)

    return out

def tanh(val: Tensor) -> Tensor:

    out = Tensor(val.d.tanh(val.data), val.device, val.dtype, (val,))

    if val.requires_grad and Tensor.grad_enabled:

        def backward():
            val.grad += (1 - out.data ** 2) * out.grad

        out._backward = backward
        out.requires_grad_(True)

    return out

def sigmoid(val: Tensor) -> Tensor:
    
        out = Tensor(val.d.sigmoid(val.data), val.device, val.dtype, (val,))
    
        if val.requires_grad and Tensor.grad_enabled:
    
            def backward():
                val.grad += out.data * (1 - out.data) * out.grad
    
            out._backward = backward
            out.requires_grad_(True)
    
        return out

def softmax(val: Tensor,dim=-1) -> Tensor:

    max_val = Tensor(val.data.max(axis=dim, keepdims=True), val.device, val.dtype)
    exp_shifted_x = (val - max_val).exp()

    sum_exp_shifted_x = exp_shifted_x.sum(dim=dim, keepdims=True)
    out = exp_shifted_x / sum_exp_shifted_x
    return out


# def cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
#     # 计算交叉熵损失
#     loss = -pred.d.sum(target.data * pred.d.log(pred.data))
    
#     out = Tensor(loss, pred.device, pred.dtype, (pred, target))

#     if pred.requires_grad and pred.grad_enabled:
#         def backward():
#             # 计算交叉熵损失的梯度
#             grad = -target.data / pred.data
#             pred.grad += grad * out.grad

#         out._backward = backward
#         out.requires_grad_(True)

#     return out
