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
            p.grad = 0

    def parameters(self):
        
        # 有梯度的张量才需要后续处理

        return [t for t in self._get_tensors() if t.requires_grad]

    def _get_tensors(self):
        # 检索Module中的所有张量
        tensors = []

        # __dict__是一个包含对象的所有属性的字典
        for _, val in self.__dict__.items():
            if isinstance(val, Tensor):
                tensors.append(val)
            elif isinstance(val, Module):
                # 如果属性是Module的子类实例,递归调用_get_tensors
                tensors.extend(val._get_tensors())
            elif isinstance(val, list):
                for element in val:
                    if isinstance(element, Module):
                        tensors.extend(element._get_tensors())

        return tensors

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
        self.W = Tensor(
            self.d.random.uniform(-1, 1, (in_features, out_features)),
            device,
            dtype,
            requires_grad=True,
        )
        if self.bias:
            self.b = Tensor(
                self.d.random.uniform(-1, 1, (1, out_features)),
                device,
                dtype,
                requires_grad=True,
            )

    def __call__(self, X: Tensor):
        out = X @ self.W + self.b
        return out
    
class Embedding(Module):
    # Embedding层，将输入的索引转换为对应的embedding向量
    # num_embeddings: 词表大小
    # embedding_dim: embedding维度
    def __init__(self, num_embeddings, embedding_dim, device="cpu", dtype="float32"):
        super().__init__(device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            self.d.random.uniform(-1, 1, (num_embeddings, embedding_dim)),
            device,
            dtype,
            requires_grad=True,
        )

    def __call__(self, indices: Tensor):
        return self.weight[indices]
    
    # def __getitem__(self, indices):
    #     return self.weight[indices]
    
class Parameter(Module):
    # 包装需要训练的参数
    def __init__(self, data, device="cpu", dtype="float32"):
        super().__init__(device)
        self.data = Tensor(data, device, dtype, requires_grad=True)

    def __call__(self):
        return self.data
    

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
