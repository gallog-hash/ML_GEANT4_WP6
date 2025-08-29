# src/core/models/activations.py

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftedSoftplus(nn.Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0):
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x):
        # softplus(x) = log(1 + exp(x)); subtract log(2) so that f(0)=0
        return F.softplus(x, beta=self.beta, threshold=self.threshold) - math.log(2)

class ELUWithLearnableOffset(nn.Module):
    def __init__(self, alpha: float = 1.0, offset_init: float = 1.0):
        super(ELUWithLearnableOffset, self).__init__()
        self.alpha = alpha
        # Initialize the offset to 1.0 so that the minimum (which is -alpha for ELU)
        # is shifted toward zero. This value can be adjusted if necessary.
        self.offset = nn.Parameter(torch.tensor(offset_init))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha) + self.offset
    
class PELU(nn.Module):
    """
    Parameterized Exponential Linear Unit (PELU).

    f(x) = (a/b) * x                     if x >= 0,
         = a * (exp(x/b) - 1)             if x < 0,
    
    where a and b are learnable parameters that adjust the scale.
    """
    def __init__(self, a_init: float = 1.0, b_init: float = 1.0):
        super(PELU, self).__init__()
        # Ensure that a and b remain positive by learning their logarithms, or
        # simply clamp them. 
        self.a = nn.Parameter(torch.tensor(a_init))
        self.b = nn.Parameter(torch.tensor(b_init))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = (self.a / self.b) * x
        neg = self.a * (torch.exp(x / self.b) - 1)
        return torch.where(x >= 0, pos, neg)