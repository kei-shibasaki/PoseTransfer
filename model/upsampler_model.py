import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math

from model.attention_block import ChanLayerNorm
from model.transformer_block import AxialTransformerBlock
from model.CPE import ConditionalPositionalEncoding
from utils import convert_bits, labels_to_bins, bins_to_labels


class ColorUpsampler(nn.Module):
    def __init__(self, opt):
        '''
        opt: option of ColorUpsampler
        '''
        super(ColorUpsampler, self).__init__()
        self.expand_dim_coarse = nn.Conv2d(3, opt['hidden_dim'], kernel_size=1, bias=False)
        self.expand_dim_source = nn.Conv2d(3, opt['hidden_dim'], kernel_size=1, bias=False)
        self.cpe_coarse = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        self.cpe_source = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        self.encoder = AxialTransformerBlock(opt['hidden_dim'], opt['heads'], opt['mlp_ratio'], opt['n_block'])
        self.to_out = nn.Sequential(
            ChanLayerNorm(opt['hidden_dim']), 
            nn.Conv2d(opt['hidden_dim'], 3, kernel_size=1, bias=False))
    
    def forward(self, coarse, source):
        coarse = self.cpe_coarse(self.expand_dim_coarse(coarse))
        source = self.cpe_source(self.expand_dim_source(source))
        output = self.encoder(coarse+source)
        output = self.to_out(output)
        return output

class SpatialUpsampler(nn.Module):
    def __init__(self, opt):
        super(SpatialUpsampler, self).__init__()
        self.expand_dim_coarse = nn.Conv2d(3, opt['hidden_dim'], kernel_size=1, bias=False)
        self.expand_dim_source = nn.Conv2d(3, opt['hidden_dim'], kernel_size=1, bias=False)
        self.cpe_coarse = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        self.cpe_source = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        self.encoder = AxialTransformerBlock(opt['hidden_dim'], opt['heads'], opt['mlp_ratio'], opt['n_block'])
        self.to_out = nn.Sequential(
            ChanLayerNorm(opt['hidden_dim']), 
            nn.Conv2d(opt['hidden_dim'], 3, kernel_size=1, bias=False))
    
    def forward(self, coarse, source):
        coarse = F.interpolate(coarse, scale_factor=4, mode='bicubic', align_corners=False)
        coarse = self.cpe_coarse(self.expand_dim_coarse(coarse))
        source = self.cpe_source(self.expand_dim_source(source))
        output = self.encoder(coarse+source)
        output = self.to_out(output)
        return output

class Upsampler(nn.Module):
    def __init__(self, opt):
        super(Upsampler, self).__init__()
        self.expand_dim_coarse = nn.Conv2d(3, opt['hidden_dim'], kernel_size=1, bias=False)
        self.expand_dim_source = nn.Conv2d(3, opt['hidden_dim'], kernel_size=1, bias=False)
        self.cpe_coarse = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        self.cpe_source = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        self.encoder = AxialTransformerBlock(opt['hidden_dim'], opt['heads'], opt['mlp_ratio'], opt['n_block'])
        self.to_out = nn.Sequential(
            ChanLayerNorm(opt['hidden_dim']), 
            nn.Conv2d(opt['hidden_dim'], 3, kernel_size=1, bias=False))
    
    def forward(self, coarse, source):
        coarse = F.interpolate(coarse, scale_factor=4, mode='bicubic', align_corners=False)
        coarse = self.cpe_coarse(self.expand_dim_coarse(coarse))
        source = self.cpe_source(self.expand_dim_source(source))
        output = self.encoder(coarse+source)
        output = self.to_out(output)
        return output
    
    def l1_loss(self, generated, target):
        b, c, h, w = generated.shape
        loss = F.l1_loss(generated, target, size_average=True)
        return loss
    
    def loss(self, generated, target):
        loss = self.l1_loss(generated, target)
        
        return loss