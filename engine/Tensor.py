import numpy as np
from typing import Tuple
from utils import broadcast_axis__,get_device

class Tensor:

    grad_enabled = True

    def __init__(
        self,
        data,
        device="cpu",
        dtype="float32",
        _prev: Tuple["Tensor"] = (),
        requires_grad=False,):
        
        self.d, self.device=get_device(device)
        self._r_grad = requires_grad

        self._backward = lambda: None
        self._prev = set([p for p in _prev if p.requires_grad])

        if isinstance(data, list) and isinstance(data[0], Tensor):
            self.stack(data)
        else:
            self.data = self.d.asarray(data, dtype=dtype)

        self.grad = (
            self.d.zeros_like(data, dtype=dtype)
            if self._r_grad
            else None
        )
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    @property
    def requires_grad(self):
        return self._r_grad
    
    def requires_grad_(self,value:bool):
        if type(value) != bool:
            raise ValueError(
                "Invalid assignment in requires_grad (only a boolean is allowed)"
            )
        if (self.grad is None) and (value):
            self.grad = self.d.zeros_like(self.data, dtype=self.dtype)
        if not value:
            self.grad = None
        self._r_grad = value

    def to(self, device: str):
        if device == self.device:
            return self

        self.d, self.device = get_device(device)
        if self.device == "cpu":
            # gpu(cupy) -> cpu(numpy)
            self.data = self.data.get()
            self.grad = (self.grad.get() if self.requires_grad and self.grad_enabled else None)
        else:
            # cpu(numpy) -> gpu(cupy)
            # now self.d is cupy
            self.data = self.d.asarray(self.data)
            self.grad = (self.d.asarray(self.grad) if self.requires_grad and self.grad_enabled else None)

        return self
    
    def __str__(self):
        device_str = f", device={self.device}" if self.device == "cuda" else ""
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        return f"tensor{repr(self.data)[5:-1]}{device_str}{grad_str})"
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return self.shape[0]
    
    def backward(self):
        if self.requires_grad == False:
            raise RuntimeError("Doesn't have a gradient (self.requires_grad is False)")

        # 对变量传播图进行拓扑排序，确保反向传播时该变量后的变量的梯度已经计算完毕
        topo = []
        visited = set()
        def build_top(v):
            if v not in visited:
                visited.add(v)
                for prev in v._prev:
                    build_top(prev)
                # 所有子节点处理完后，将自己加入到topo中，即topo中任意节点都在其子节点的后面
                topo.append(v)

        build_top(self)

        # 当前变量的梯度为1
        self.grad = self.d.ones_like(self.grad)
        for node in reversed(topo):
            node._backward()
    
    def stack(self, tensors, axis=0):
        if not all([isinstance(t, Tensor) for t in tensors]):
            raise ValueError("All elements must be of type Tensor")
        if not all([t.device == tensors[0].device for t in tensors]):
            raise ValueError("All elements must be on the same device")
        if not all([t.dtype == tensors[0].dtype for t in tensors]):
            raise ValueError("All elements must be of the same dtype")
        self.data = self.d.stack([t.data for t in tensors], axis=axis)
        self.device = tensors[0].device
        # self.dtype = tensors[0].dtype
        self._prev = set([p for p in tensors if p.requires_grad])
        # self.shape = self.data.shape
        self._r_grad=(any([t.requires_grad for t in tensors]))
        if self.requires_grad and self.grad_enabled:
            def backward():
                self.grad = self.grad.swapaxes(axis, 0)
                for i, t in enumerate(tensors):
                    if t.requires_grad:
                        t.grad += self.grad[i]
            self._backward = backward


    def clip(
        self,
        min=1e-7,
        max=1 - 1e-7,
        clip_grad=False,
        g_min=1e-7,
        g_max=1 - 1e-7,
    ):
        #在指定范围内裁剪张量值和梯度

        self.data = self.data.clip(min, max)

        if clip_grad and (self.grad != None):
            self.grad = self.grad.clip(g_min, g_max)

        return self

    def _check(
        self, other, raise_error_right_away=False, if_tensor=False, if_same_device=False
    ):
        if if_tensor and (not isinstance(other, Tensor)):
            raise_error_right_away = True
            error_str = "Unsupported datatype; Expected a Tensor."

        elif if_same_device and (self.device != other.device):
            raise_error_right_away = True
            error_str = "Expected all tensors to be on the same device, but found at least two devices, cuda and cpu!"

        if raise_error_right_away:
            raise RuntimeError(error_str)


    def __getitem__(self, index):
        if isinstance(index, (slice, int)):
            # 处理切片操作或整数索引操作
            out = Tensor(self.data[index], self.device, self.dtype,_prev=(self,))
            if self.requires_grad and self.grad_enabled:
                def backward():
                    if isinstance(index, slice):
                        # 处理切片操作的反向传播
                        grad_slice = self.grad[index] 
                        grad_slice += out.grad
                    else:
                        # 处理整数索引操作的反向传播
                        self.grad[index] += out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        if isinstance(index, tuple):
            # 处理多维索引操作
            out_data = self.data[index]
            out = Tensor(out_data, self.device, self.dtype, _prev=(self,))
            if self.requires_grad and self.grad_enabled:
                def backward():
                    self.grad[index] += out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        elif isinstance(index, list):
            # 处理列表索引操作
            out_data = self.data[index]
            out = Tensor(out_data, self.device, self.dtype, _prev=(self,))
            if self.requires_grad and self.grad_enabled:
                def backward():
                    self.grad[index] += out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        elif isinstance(index, Tensor):
            # 处理张量索引操作
            # data = index.data.copy()
            out_data = self.data[index.data.astype(int)]
            out = Tensor(out_data, self.device, self.dtype, _prev=(self,))
            if self.requires_grad and self.grad_enabled:
                def backward():
                    # 处理张量索引操作的反向传播
                    self.grad[index.data.astype(int)] += out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        elif isinstance(index,self.d.ndarray):
            # 处理numpy/cupy数组索引操作
            out_data = self.data[index.astype(int)]
            out = Tensor(out_data, self.device, self.dtype, _prev=(self,))
            if self.requires_grad and self.grad_enabled:
                def backward():
                    self.grad[index.astype(int)] += out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        else:
            self._check(None, raise_error_right_away=True)

    # 改变张量的形状
    def reshape(self, *shape):
        
        out=Tensor(self.data.reshape(shape), self.device, self.dtype, _prev=(self,))
        if(self.requires_grad and self.grad_enabled):
            def backward():
                self.grad += out.grad.reshape(self.shape)
            out._backward = backward
            out.requires_grad_(True)
        return out
    

    def masked_fill(self, mask, value):
        # mask 是一个布尔numpy或cupy数组，value 是一个标量
        data = self.data.copy()
            
        # 在 mask 的前面扩展维度
        mask_expanded = self.d.broadcast_to(mask, self.data.shape)
        
        data[mask_expanded] = value
        out = Tensor(data, self.device, self.dtype, _prev=(self,))
        if self.requires_grad and self.grad_enabled:
            def backward():
                grad = out.grad.copy()
                grad[mask_expanded] = 0
                self.grad += grad
            out._backward = backward
            out.requires_grad_(True)
        return out


    @property
    def T(self):
        # 转置操作
        out=Tensor(self.data.T, self.device, self.dtype, _prev=(self,))
        if(self.requires_grad and self.grad_enabled):
            def backward():
                self.grad += out.grad.T
            out._backward = backward
            out.requires_grad_(True)
        return out
    
    # 转置操作
    def transpose(self, *axes):
        
        if len(axes) == 0:
            # 如果没有提供参数,则执行常规转置操作
            out = Tensor(self.data.T, self.device, self.dtype, _prev=(self,))
        elif len(axes) == 2:
            # 如果提供了两个参数,则按照指定的轴进行转置
            axis1, axis2 = axes
            out = Tensor(self.data.swapaxes(axis1, axis2), self.device, self.dtype, _prev=(self,))
        else:
            raise ValueError("Invalid number of arguments. Expected 0 or 2 arguments.")

        if self.requires_grad and self.grad_enabled:
            def backward():
                if len(axes) == 0:
                    self.grad += out.grad.T
                else:
                    self.grad += out.grad.swapaxes(axis2, axis1)
            out._backward = backward
            out.requires_grad_(True)

        return out
    
    def view(self, *shape):
        # 改变张量的形状
        out = Tensor(self.data.reshape(shape), self.device, self.dtype, _prev=(self,))
        if self.requires_grad and self.grad_enabled:
            def backward():
                self.grad += out.grad.reshape(self.shape)
            out._backward = backward
            out.requires_grad_(True)
        return out

    # 沿着指定轴求和，未指定维度时全部求和，keepdims表示是否保留维度
    def sum(self, dim=None, keepdims=False, dtype=None):

        sum_val = self.d.sum(self.data, axis=dim, keepdims=keepdims, dtype=dtype)
        out = Tensor(sum_val, self.device, self.dtype, (self,))

        if self.requires_grad and self.grad_enabled:
            expand_dims = dim if dim and not keepdims else ()

            # 需要恢复out被sum操作压缩的维度，以便广播计算梯度
            def backward():
                self.grad += self.d.ones_like(self.grad) * self.d.expand_dims(
                    out.grad, axis=expand_dims
                )

            out._backward = backward
            out.requires_grad_(True)

        return out

    # 沿着指定轴拼接张量，未指定维度时按照第一个维度拼接
    @staticmethod
    def cat(tensors, dim=0):
        if not all(isinstance(t, Tensor) for t in tensors):
            raise ValueError("All elements must be of type Tensor")
        if not all(t.device == tensors[0].device for t in tensors):
            raise ValueError("All elements must be on the same device")
        if not all(t.dtype == tensors[0].dtype for t in tensors):
            raise ValueError("All elements must be of the same dtype")
        
        out = Tensor(np.concatenate([t.data for t in tensors], axis=dim), tensors[0].device, tensors[0].dtype)
        
        if any(t.requires_grad for t in tensors) and Tensor.grad_enabled:
            def backward():
                start = 0
                for t in tensors:
                    if t.requires_grad:
                        t.grad += out.grad.take(range(start, start + t.shape[dim]), axis=dim)
                    start += t.shape[dim]
            out._backward = backward
            out.requires_grad_(True)
        
        return out

    # 沿着指定轴求平均值，未指定维度时全部求平均值，keepdims表示是否保留维度
    def mean(self, dim=None, keepdims=False, dtype=None):
            
            mean_val = self.d.mean(self.data, axis=dim, keepdims=keepdims, dtype=dtype)
            out = Tensor(mean_val, self.device, self.dtype, (self,))
    
            if self.requires_grad and self.grad_enabled:
                expand_dims = dim if dim and not keepdims else ()
    
                def backward():
                    self.grad += self.d.ones_like(self.grad) * self.d.expand_dims(
                        out.grad, axis=expand_dims
                    )
    
                out._backward = backward
                out.requires_grad_(True)
    
            return out
    
    # 沿着指定轴求方差，未指定维度时全部求方差，keepdims表示是否保留维度
    def var(self, dim=None, keepdims=False, dtype=None):
        
        var_val = self.d.var(self.data, axis=dim, keepdims=keepdims, dtype=dtype)
        out = Tensor(var_val, self.device, self.dtype, (self,))

        if self.requires_grad and self.grad_enabled:
            expand_dims = dim if dim and not keepdims else ()

            def backward():
                self.grad += self.d.expand_dims(
                    out.grad, axis=expand_dims
                ) * 2 * (self.data - self.d.expand_dims(out.data, axis=expand_dims)) / self.data.size

            out._backward = backward
            out.requires_grad_(True)

        return out

    def exp(self):
        out=Tensor(self.d.exp(self.data), self.device, self.dtype, _prev=(self,))
        if(self.requires_grad and self.grad_enabled):
            def backward():
                self.grad += out.grad*out.data
            out._backward = backward
            out.requires_grad_(True)
        return out
    
    def log(self):
        out=Tensor(self.d.log(self.data), self.device, self.dtype, _prev=(self,))
        if(self.requires_grad and self.grad_enabled):
            def backward():
                eps = 1e-8  # 添加一个小常数，避免除以零
                self.grad += out.grad / (self.data + eps)
            out._backward = backward
            out.requires_grad_(True)
        return out
    

    # 重载加法运算符
    def __add__(self, other):
        if isinstance(other, (int, float)):
            # other的形状会自动广播
            out = Tensor(self.data+other, self.device, self.dtype, _prev=(self,))
            if(self.requires_grad and self.grad_enabled):
                def backward():
                    self.grad += out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        
        elif isinstance(other, Tensor):
            self._check(other, if_same_device=True)
            out=Tensor(self.data+other.data, self.device, self.dtype, _prev=(self, other))

            # 如果两个张量都不关心梯度计算，直接返回结果
            if self.requires_grad == other.requires_grad == False:
                return out
            if self.grad_enabled:
                if self.shape == other.shape:
                    # 如果两个张量的形状相同，直接计算梯度
                    def backward():
                        if self.requires_grad:
                            self.grad += 1*out.grad
                        if other.requires_grad:
                            other.grad += 1*out.grad
                else:
                    # 如果两个张量的形状不相同，需要广播梯度进行计算，值的计算已经在numpy中广播计算完成，out的形状为广播后的形状
                    axis_self, axis_other = broadcast_axis__(self.shape, other.shape)
                    def backward():
                        # 将out的梯度形状转换到self和other的形状
                        if self.requires_grad:
                            self.grad += out.grad.sum(axis=axis_self).reshape(self.shape)
                        if other.requires_grad:
                            other.grad += out.grad.sum(axis=axis_other).reshape(other.shape)
                out._backward = backward
                out.requires_grad_(True)
                return out
        else:
            self._check(None, raise_error_right_away=True)

    # 重载乘法运算符,按位乘法
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Tensor(self.data*other, self.device, self.dtype, _prev=(self,))
            if(self.requires_grad and self.grad_enabled):
                def backward():
                    self.grad += other*out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        
        elif isinstance(other, Tensor):
            self._check(other, if_same_device=True)
            out=Tensor(self.data*other.data, self.device, self.dtype, _prev=(self, other))

            if self.requires_grad == other.requires_grad == False:
                return out
            if self.grad_enabled:
                if self.shape == other.shape:
                    def backward():
                        if self.requires_grad:
                            self.grad += other.data*out.grad
                        if other.requires_grad:
                            other.grad += self.data*out.grad
                else:
                    axis_self, axis_other = broadcast_axis__(self.shape, other.shape)
                    def backward():
                        if self.requires_grad:
                            self.grad += (out.grad*other.data).sum(axis=axis_self).reshape(self.shape)
                        if other.requires_grad:
                            # print((out.grad*self.data).shape)
                            # print((out.grad*self.data).sum(axis=axis_other).shape)
                            # print(other.shape)
                            other.grad += (out.grad*self.data).sum(axis=axis_other).reshape(other.shape)
                out._backward = backward
                out.requires_grad_(True)
                return out
        else:
            self._check(None, raise_error_right_away=True)

    # 矩阵乘法
    def __matmul__(self, other):
        self._check(other, if_same_device=True)
        self._check(other, if_tensor=True)
        out = Tensor(self.data @ other.data, self.device, self.dtype, (self, other))
        
        if self.requires_grad == other.requires_grad == False:
            return out

        if self.grad_enabled:
            if self.data.ndim == other.data.ndim == 2:
                # 2维矩阵乘法
                def backward():
                    # C=A@B,  dC/dA = dC/dC * dC/dA = dC/dC * B.T
                    # C=A@B,  dC/dB = dC/dC * dC/dB = A.T * dC/dC
                    if self.requires_grad:
                        self.grad += out.grad @ other.data.T
                    if other.requires_grad:
                        other.grad += self.data.T @ out.grad
            else:
                if self.data.ndim == 1:
                    self_expand_axis = (0,)
                    self_expanded_shape = (1,) + self.shape
                else:
                    self_expand_axis = ()
                    self_expanded_shape = self.shape

                if other.data.ndim == 1:
                    other_expand_axis = (-1,)
                    other_expanded_shape = (1,) + other.shape
                else:
                    other_expand_axis = ()
                    other_expanded_shape = other.shape

                result_expand_axis = self_expand_axis + other_expand_axis
                
                # 确定 self 和 other 在批次维度上的广播轴 axis_self 和 axis_other
                axis_self, axis_other = broadcast_axis__(
                    self_expanded_shape[:-2], other_expanded_shape[:-2]
                )

                def backward():
                    if self.requires_grad:
                        self.grad += self.d.reshape(
                            self.d.sum(
                                self.d.squeeze(
                                    self.d.expand_dims(out.grad, axis=result_expand_axis) @ self.d.expand_dims(other.data, axis=other_expand_axis).swapaxes(-1, -2),
                                    axis=self_expand_axis,
                                ),
                                axis=axis_self,
                            ),
                            self.shape,
                        )

                    if other.requires_grad:
                        other.grad += self.d.reshape(
                            self.d.sum(
                                self.d.squeeze(
                                    self.d.expand_dims(self.data, axis=self_expand_axis).swapaxes(-1, -2) @ self.d.expand_dims(out.grad, axis=result_expand_axis),
                                    axis=other_expand_axis,
                                ),
                                axis=axis_other,
                            ),
                            other.shape,
                        )
            out._backward = backward
            out.requires_grad_(True)
        
        return out

    # 幂运算
    def __pow__(self, other):

        def _neg_pow(a, b):
            return 1 / (a ** self.d.abs(b)) if b < 0 else a**b

        if isinstance(other, (int, float)):
            out = Tensor(_neg_pow(self.data, other), self.device, self.dtype, (self,))
            if(self.requires_grad and self.grad_enabled):
                def backward():
                    self.grad += other * _neg_pow(self.data, other - 1) * out.grad
                out._backward = backward
                out.requires_grad_(True)
            return out
        else:
            self._check(None, raise_error_right_away=True)

    # 加法交换
    def __radd__(self, other):
        return self + other

    # 乘法交换
    def __rmul__(self, other):
        return self * other

    # 除法
    def __truediv__(self, other):
        return self * (other**-1)

    # 除法交换
    def __rtruediv__(self, other):
        return (self**-1) * other

    # 取负
    def __neg__(self):
        return self * -1

    # 减法
    def __sub__(self, other):
        return self + (-other)

    # 减法交换
    def __rsub__(self, other):
        return -self + other