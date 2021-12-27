import torch
from torch import nn
import numpy as np

from model.attention_block import *

class SourcePoseEncoder(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, n_block):
        super(SourcePoseEncoder, self).__init__()
        self.n_block = n_block
        
        for i in range(n_block):
            layernorm = ChanLayerNorm(dim)
            row_attn = AxialAttention(dim, heads, mode='row', masked=False)
            col_attn = AxialAttention(dim, heads, mode='colmun', masked=False)
            mlp = MLP(dim, mlp_ratio)
            setattr(self, f'layernorm_{i}', layernorm)
            setattr(self, f'row_attn_{i}', row_attn)
            setattr(self, f'col_attn_{i}', col_attn)
            setattr(self, f'mlp_{i}', mlp)
    
    def forward(self, x):
        for i in range(self.n_block):
            layernorm = getattr(self, f'layernorm_{i}')
            row_attn = getattr(self, f'row_attn_{i}')
            col_attn = getattr(self, f'col_attn_{i}')
            mlp = getattr(self, f'mlp_{i}')
            
            a = x
            x = layernorm(x)
            x = row_attn(x)
            x = col_attn(x)
            x = a + x
            x = mlp(x)

