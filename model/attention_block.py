import torch
from torch import nn
import numpy as np

### Not Conditional ###
class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class AxialAttention(nn.Module):
    def __init__(self, dim, heads, mode, masked=False):
        super(AxialAttention, self).__init__()
        assert mode in ['row', 'column']
        assert dim%heads==0
        self.mode = mode
        self.dim = dim
        self.heads = heads
        self.dim_heads = dim//heads
        self.masked = masked
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2*dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
    
    def create_mask(self, x):
        # input shape: (BH, W, C) or (BW, H, C)
        b, l, _ = x.shape
        mask = np.tril(np.ones((b, l, l)), k=0).astype('uint8')
        return torch.Tensor(mask).int()
    
    def forward(self, x, kv=None):
        b_size, channel, height, width = x.shape
        if self.mode=='row':
            x = x.reshape(-1, width, channel)
        else: # column
            x = x.reshape(-1, height, channel)
        
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2,dim=-1))
        
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b*h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))
        
        dots = torch.einsum('b i e, b j e -> b i j', q, k) * (e**-0.5)
        
        if self.masked:
            mask = self.create_mask(dots)
            dots = dots.masked_fill(mask==0, -1e9)
        
        dots = dots.softmax(dim=-1)
        out = torch.einsum('b i j, b j e -> b i e', dots, v)
        
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        
        out = out.reshape(b_size, channel, height, width)
        
        return out

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.lnorm = ChanLayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim*mlp_ratio, kernel_size=1, bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(dim*mlp_ratio, dim, kernel_size=1, bias=False)
    
    def forward(self, x):
        h = self.lnorm(x)
        h = self.conv1(h)
        h = self.gelu(h)
        h = self.conv2(h)
        return x + h


### Conditional ###
class ConditionalChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super(ConditionalChanLayerNorm, self).__init__()
        self.eps = eps
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.to_g = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_b = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x, context):
        assert x.shape==context.shape
        g = self.to_g(self.gap(context))
        b = self.to_b(self.gap(context))
        
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x-mean) / (std+self.eps) * g + b

class ConditionalAxialAttention(nn.Module):
    def __init__(self, dim, heads, mode, masked=False):
        super(ConditionalAxialAttention, self).__init__()
        assert mode in ['row', 'column']
        assert dim%heads==0
        self.mode = mode
        self.dim = dim
        self.heads = heads
        self.dim_heads = dim//heads
        self.masked = masked
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, 2*dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.to_scale = nn.Linear(dim, dim, bias=False)
        self.to_shift = nn.Linear(dim, dim, bias=False)
    
    def create_mask(self, x):
        # input shape: (BH, W, C) or (BW, H, C)
        b, l, _ = x.shape
        mask = np.tril(np.ones((b, l, l)), k=0).astype('uint8')
        return torch.Tensor(mask).int()
    
    def forward(self, x, context, kv=None):
        assert x.shape==context.shape
        b_size, channel, height, width = x.shape
        if self.mode=='row':
            x = x.reshape(-1, width, channel)
            context = context.reshape(-1, width, channel)
        else: # column
            x = x.reshape(-1, height, channel)
            context = context.reshape(-1, height, channel)
        
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2,dim=-1))
        scale = self.to_scale(context)
        shift = self.to_shift(context)
        
        q = scale*q + shift
        k = scale*k + shift
        v = scale*v + shift
        
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b*h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))
        
        dots = torch.einsum('b i e, b j e -> b i j', q, k) * (e**-0.5)
        
        if self.masked:
            mask = self.create_mask(dots)
            dots = dots.masked_fill(mask==0, -1e9)
        
        dots = dots.softmax(dim=-1)
        out = torch.einsum('b i j, b j e -> b i e', dots, v)
        
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        
        out = out.reshape(b_size, channel, height, width)
        
        return out

class ConditionalMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super(ConditionalMLP, self).__init__()
        self.clnorm = ConditionalChanLayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim*mlp_ratio, kernel_size=1, bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(dim*mlp_ratio, dim, kernel_size=1, bias=False)
        
        self.to_scale = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_shift = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
    
    def forward(self, x, context):
        assert x.shape==context.shape
        h = self.clnorm(x, context)
        h = self.conv1(h)
        h = self.gelu(h)
        h = self.conv2(h)
        h = x + h
        
        scale = self.to_scale(context)
        shift = self.to_shift(context)
        return scale*h + shift