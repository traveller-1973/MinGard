from warnings import warn
import numpy as np
from __init__ import get_d__


class Optimizer:
    def __init__(self, parameters, lr, device: str):
        """
        Base class for optimizers.

        Args:
            parameters (list): List of tensors representing the trainable parameters.
            lr (float): Learning rate for optimization.
            device (str): Device on which the optimizer's computations should be performed.
        """
        self.d, self.device = get_d__(device)
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Update parameters"""
        raise NotImplementedError

    def zero_grad(self):
        """
        Reset gradients of all parameters to zero.
        """
        for p in self.parameters:
            p.grad = self.d.zeros_like(p.grad)
            
class SGD(Optimizer):
    def __init__(self, parameters, lr, momentum=0.9, device="cpu"):
        """
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters (list): List of tensors representing the trainable parameters.
            lr (float): Learning rate for optimization.
            momentum (float, optional): Momentum factor. Default is 0.9.
            device (str, optional): Device on which the optimizer's computations should be performed. Default is "cpu".
        """
        super().__init__(parameters, lr, device)
        self.momentum = momentum
        self.velocities = [self.d.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self.velocities):
            v[:] = ((1 - self.momentum) * p.grad) + (self.momentum * v)
            p.data -= self.lr * v