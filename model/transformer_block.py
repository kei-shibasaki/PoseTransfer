from model.CPE import ConditionalPositionalEncoding
import torch
from torch import nn
import numpy as np

from model.attention_block import *
from model.other_block import Shift

### Pose Transfer Model ###
class SourcePoseEncoder(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, n_block):
        super(SourcePoseEncoder, self).__init__()
        self.n_block = n_block
        
        for i in range(n_block):
            layernorm = ChanLayerNorm(dim)
            row_attn = AxialAttention(dim, heads, mode='row', masked=False)
            col_attn = AxialAttention(dim, heads, mode='column', masked=False)
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
        return x

class OuterDecoder(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, n_block):
        super(OuterDecoder, self).__init__()
        self.n_block = n_block
        
        for i in range(n_block):
            clayernorm = ConditionalChanLayerNorm(dim)
            crow_attn = ConditionalAxialAttention(dim, heads, mode='row', masked=False)
            ccol_attn = ConditionalAxialAttention(dim, heads, mode='column', masked=True)
            cmlp = ConditionalMLP(dim, mlp_ratio)
            setattr(self, f'clayernorm_{i}', clayernorm)
            setattr(self, f'crow_attn_{i}', crow_attn)
            setattr(self, f'ccol_attn_{i}', ccol_attn)
            setattr(self, f'cmlp_{i}', cmlp)
        
        self.shift_down = Shift('down')
    
    def forward(self, x, context):
        for i in range(self.n_block):
            clayernorm = getattr(self, f'clayernorm_{i}')
            crow_attn = getattr(self, f'crow_attn_{i}')
            ccol_attn = getattr(self, f'ccol_attn_{i}')
            cmlp = getattr(self, f'cmlp_{i}')
            
            a = x
            x = clayernorm(x, context)
            x = crow_attn(x, context)
            x = ccol_attn(x, context)
            x = a + x
            x = cmlp(x, context)
        
        x = self.shift_down(x)
        return x

class InnerDecoder(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, n_block):
        super(InnerDecoder, self).__init__()
        self.n_block = n_block
        self.shift_right = Shift('right')
        
        for i in range(n_block):
            clayernorm = ConditionalChanLayerNorm(dim)
            crow_attn = ConditionalAxialAttention(dim, heads, mode='row', masked=True)
            cmlp = ConditionalMLP(dim, mlp_ratio)
            setattr(self, f'clayernorm_{i}', clayernorm)
            setattr(self, f'crow_attn_{i}', crow_attn)
            setattr(self, f'cmlp_{i}', cmlp)
    
    def forward(self, x, upper_context, channel_context):
        x = self.shift_right(x)
        context = upper_context + channel_context
        x = x + upper_context
        for i in range(self.n_block):
            clayernorm = getattr(self, f'clayernorm_{i}')
            crow_attn = getattr(self, f'crow_attn_{i}')
            cmlp = getattr(self, f'cmlp_{i}')
            
            a = x
            x = clayernorm(x, context)
            x = crow_attn(x, context)
            x = a + x
            x = cmlp(x, context)
        return x


### Upsampler ###
class AxialTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, n_block):
        super(AxialTransformerBlock, self).__init__()
        self.n_block = n_block
        
        for i in range(n_block):
            layernorm = ChanLayerNorm(dim)
            row_attn = AxialAttention(dim, heads, mode='row', masked=False)
            col_attn = AxialAttention(dim, heads, mode='column', masked=False)
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
        return x