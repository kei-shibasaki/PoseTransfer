import torch
from torch import nn
import numpy as np

from model_simple.transformer_block import Encoder, Decoder
from model_simple.CPE import ConditionalPositionalEncoding

class SimplePoseTransferModel(nn.Module):
    def __init__(self):
        super(SimplePoseTransferModel, self).__init__()
        self.expand_dim1 = nn.Conv2d(3+18, 64, kernel_size=1, bias=False)
        self.expand_dim2 = nn.Conv2d(18, 64, kernel_size=1, bias=False)
        self.cpe1 = ConditionalPositionalEncoding(64)
        self.cpe2 = ConditionalPositionalEncoding(64)
        
        self.enc1 = Encoder(64, heads=8, mlp_ratio=4, n_block=2)
        self.enc2 = Encoder(64, heads=8, mlp_ratio=4, n_block=2)
        
        self.dec = Decoder(64, heads=8, mlp_ratio=4, n_block=2)
        self.to_out = nn.Conv2d(64, 3, kernel_size=1, bias=False)
    
    def forward(self, P1, map1, map2):
        q = self.cpe1(self.expand_dim1(torch.cat([P1, map1], dim=1)))
        kv = self.cpe2(self.expand_dim2(map2))
        
        q = self.enc1(q)
        kv = self.enc2(kv)
        
        out = self.dec(q, kv)
        out = self.to_out(out)
        return torch.tanh(out)
        