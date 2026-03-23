import pickle
from collections import OrderedDict
from typing import Optional, Set
import numpy as np

from engine.Tensor import Tensor
from utils import get_device


class Module:
    """
    Base class for neural network modules.

    Args:
        device (str): Device on which the module's tensors should reside.
    """

    def __init__(self, device="cpu"):
        self.d, self.device = get_device(device)

    def zero_grad(self):
        """
        Reset gradients of all tensors in the module to zero.
        """
        for p in self.parameters():
            if p.grad is not None:
                p.grad.fill(0)

    def parameters(self):
        """
        Return a list of tensors that require gradients.
        """
        return [tensor for _, tensor in self.named_parameters()]

    def named_parameters(self, prefix: str = ""):
        """
        Yield tuples of (name, tensor) for all parameters that require gradients.
        """
        yield from (
            (name, tensor)
            for name, tensor in self._named_tensors(prefix=prefix)
            if tensor.requires_grad
        )

    def named_children(self):
        """Yield tuples of (name, child_module) for all direct submodules."""
        memo: Set[int] = set()
        for attr_name, value in self.__dict__.items():
            if attr_name in {"d", "device"}:
                continue

            if isinstance(value, Module):
                if id(value) in memo:
                    continue
                memo.add(id(value))
                yield attr_name, value
            elif isinstance(value, (list, tuple)):
                for idx, element in enumerate(value):
                    if isinstance(element, Module):
                        if id(element) in memo:
                            continue
                        memo.add(id(element))
                        yield f"{attr_name}.{idx}", element
            elif isinstance(value, dict):
                for key, element in value.items():
                    if isinstance(element, Module):
                        if id(element) in memo:
                            continue
                        memo.add(id(element))
                        yield f"{attr_name}.{key}", element

    def children(self):
        """Yield direct child modules."""
        for _, module in self.named_children():
            yield module

    def extra_repr(self) -> str:
        """Return the string displayed in the module repr after the name."""
        return ""

    def __repr__(self) -> str:  # pragma: no cover - representation helper
        lines = []
        children = list(self.named_children())
        extra_repr = self.extra_repr()
        if extra_repr:
            lines.extend(extra_repr.split("\n"))

        for name, child in children:
            child_repr = repr(child)
            child_repr = self._addindent(child_repr, 2)
            lines.append(f"({name}): {child_repr}")

        main_str = f"{self.__class__.__name__}("
        if children:
            main_str += "\n  " + "\n  ".join(lines) + "\n"
        elif lines:
            if len(lines) == 1:
                main_str += lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    @staticmethod
    def _addindent(text: str, num_spaces: int) -> str:
        lines = text.split("\n")
        if len(lines) == 1:
            return text
        indent = " " * num_spaces
        return lines[0] + "\n" + "\n".join(indent + line for line in lines[1:])

    def _named_tensors(self, memo: Optional[Set[int]] = None, prefix: str = ""):
        if memo is None: #防止权重共享时重复访问同一Tensor
            memo = set()

        for attr_name, value in self.__dict__.items():
            if attr_name in {"d", "device"}:
                continue

            name = f"{prefix}{attr_name}" if prefix else attr_name

            if isinstance(value, Tensor):
                if id(value) in memo:
                    continue
                memo.add(id(value))
                yield name, value
            elif isinstance(value, Module):
                yield from value._named_tensors(memo, prefix=f"{name}.")
            elif isinstance(value, (list, tuple)):
                for idx, element in enumerate(value):
                    indexed_name = f"{name}.{idx}"
                    if isinstance(element, Tensor):
                        if id(element) in memo:
                            continue
                        memo.add(id(element))
                        yield indexed_name, element
                    elif isinstance(element, Module):
                        yield from element._named_tensors(
                            memo, prefix=f"{indexed_name}."
                        )
            elif isinstance(value, dict):
                for key, element in value.items():
                    indexed_name = f"{name}.{key}"
                    if isinstance(element, Tensor):
                        if id(element) in memo:
                            continue
                        memo.add(id(element))
                        yield indexed_name, element
                    elif isinstance(element, Module):
                        yield from element._named_tensors(
                            memo, prefix=f"{indexed_name}."
                        )

    def _get_tensors(self):
        return [tensor for _, tensor in self._named_tensors()]

    def state_dict(self, keep_vars: bool = False):
        """
        Return an OrderedDict containing copies of the module's parameters.
        """
        state = OrderedDict()
        for name, tensor in self._named_tensors():
            state[name] = tensor if keep_vars else self._tensor_to_numpy(tensor)
        return state

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load parameters from *state_dict* into current module."""
        name_to_tensor = dict(self._named_tensors())
        missing_keys = []
        unexpected_keys = []

        for name, tensor in name_to_tensor.items():
            if name not in state_dict:
                if strict:
                    missing_keys.append(name)
                continue

            param = state_dict[name]
            array = self._to_numpy_array(param)

            if tensor.shape != array.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{name}': "
                    f"expected {tensor.shape}, got {array.shape}."
                )

            tensor.data = tensor.d.asarray(array, dtype=tensor.dtype)
            if tensor.grad is not None:
                tensor.grad = tensor.d.zeros_like(tensor.data)

        if strict:
            unexpected_keys = [name for name in state_dict.keys() if name not in name_to_tensor]
            if missing_keys or unexpected_keys:
                raise KeyError(
                    "Error(s) in loading state_dict: "
                    + (f"Missing keys: {missing_keys}. " if missing_keys else "")
                    + (f"Unexpected keys: {unexpected_keys}." if unexpected_keys else "")
                )

    @staticmethod
    def _tensor_to_numpy(tensor: Tensor):
        data = tensor.data
        if hasattr(data, "get"):
            data = data.get()
        return np.array(data, copy=True)

    @staticmethod
    def _to_numpy_array(value):
        if isinstance(value, Tensor):
            value = value.data
        if hasattr(value, "get"):
            value = value.get()
        return np.array(value, copy=False)

    def to(self, device: str):

        if device == self.device:
            return self

        self.d, self.device = get_device(device)

        for tensor in self._get_tensors():
            tensor.to(device)

        return self
    

class Linear(Module):

    # 线性层，y=wx+b，其中w是权重，b是偏置

    def __init__(
        self, in_features, out_features, bias=True, device="cpu", dtype="float32"
    ):
        super().__init__(device)
        self.bias = bias
        limit = (1 / in_features) ** 0.5
        self.W = Tensor(
            self.d.random.uniform(-limit, limit, (in_features, out_features)),
            device,
            dtype,
            requires_grad=True,
        )
        if self.bias:
            self.b = Tensor(
                self.d.random.uniform(-limit, limit, (1, out_features)),
                device,
                dtype,
                requires_grad=True,
            )

    def __call__(self, X: Tensor):
        out = X @ self.W
        if self.bias:
            out += self.b
        return out

    def extra_repr(self) -> str:
        in_features, out_features = self.W.shape
        return (
            f"in_features={in_features}, out_features={out_features}, bias={self.bias}"
        )
    
class Embedding(Module):
    # Embedding层，将输入的索引转换为对应的embedding向量
    # num_embeddings: 词表大小
    # embedding_dim: embedding维度
    def __init__(self, num_embeddings, embedding_dim, device="cpu", dtype="float32"):
        super().__init__(device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        limit = (1 / embedding_dim) ** 0.5
        self.weight = Tensor(
            self.d.random.uniform(-limit, limit, (num_embeddings, embedding_dim)),
            device,
            dtype,
            requires_grad=True,
        )

    def __call__(self, indices: Tensor):
        return self.weight[indices]

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"
        )
    
    # def __getitem__(self, indices):
    #     return self.weight[indices]
    
class Parameter(Module):
    # 包装需要训练的参数
    def __init__(self, data, device="cpu", dtype="float32"):
        super().__init__(device)
        self.data = Tensor(data, device, dtype, requires_grad=True)

    def __call__(self):
        return self.data

    def extra_repr(self) -> str:
        return f"shape={self.data.shape}, dtype={self.data.dtype}"
    

class ModuleList(Module):
    # ModuleList，用于包装多个Module
    def __init__(self, modules, device="cpu"):
        super().__init__(device)
        self.modules = modules

    def __call__(self, x):
        for module in self.modules:
            x = module(x)
        return x
    
    def __iter__(self):
        return iter(self.modules)
    
    def named_children(self):
        seen: Set[int] = set()
        for idx, module in enumerate(self.modules):
            if not isinstance(module, Module):
                continue
            if id(module) in seen:
                continue
            seen.add(id(module))
            yield str(idx), module

    def _get_tensors(self):
        tensors = []
        for module in self.modules:
            tensors.extend(module._get_tensors())
        return tensors


def save_state_dict(module: Module, file_path: str):
    """Persist the module's parameters to disk."""
    state = module.state_dict()
    with open(file_path, "wb") as f:
        pickle.dump(state, f)


def load_state_dict(module: Module, file_path: str, strict: bool = True):
    """Load module parameters from *file_path* into *module*."""
    with open(file_path, "rb") as f:
        state = pickle.load(f)
    module.load_state_dict(state, strict=strict)

