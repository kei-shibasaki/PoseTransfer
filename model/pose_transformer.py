import torch
from torch import nn
from torch.nn import functional as F

from model.axial_transformer import AxialTransformerBlock, ChanLayerNorm
from model.positional_encoding import ConditionalPositionalEncoding

class CNNTransformation(nn.Module):
    def __init__(self, opt, idx):
        super(CNNTransformation, self).__init__()
        depth = opt.cnn_depths[idx]
        dim = opt.cnn_dims[idx]
        in_channels = opt.cnn_in_channels[idx]
        ksize = opt.cnn_ksizes[idx]
        
        layers = []
        for i in range(depth):
            if i==0:
                layers = layers + [
                    nn.Conv2d(in_channels, dim, kernel_size=ksize, padding=ksize//2), 
                    nn.ReLU(inplace=True), 
                    ChanLayerNorm(dim)]
            else:
                layers = layers + [
                    nn.Conv2d(dim, dim, kernel_size=ksize, padding=ksize//2), 
                    nn.ReLU(inplace=True), 
                    ChanLayerNorm(dim)]
            self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class ATBTransformation(nn.Module):
    def __init__(self, opt, idx):
        super(ATBTransformation, self).__init__()
        depth = opt.atb_depths[idx]
        dim = opt.atb_dims[idx]
        heads = opt.atb_heads[idx]
        mlp_ratio = opt.atb_mlp_ratios[idx]
        drop = opt.atb_drops[idx]
        in_channels = opt.atb_in_channels[idx]
        cpe_ksize = opt.atb_cpe_ksizes[idx]
        
        self.expand_dim = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.cpe = ConditionalPositionalEncoding(dim, cpe_ksize)
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = AxialTransformerBlock(
                dim, heads=heads, mlp_ratio=mlp_ratio, dropout=drop)
            self.layers.append(layer)
    
    def forward(self, x):
        x = self.cpe(self.expand_dim(x))
        for layer in self.layers:
            x = layer(x)
        return x

### Main Model ###
class PoseTransformer(nn.Module):
    def __init__(self, opt):
        '''
        Args:
            opt class Config: See details in config/config.py
        '''
        super(PoseTransformer, self).__init__()
        self.n_cnn_trans = opt.n_cnn_trans
        self.n_atb_trans = opt.n_atb_trans
        
        self.mean = torch.tensor((0.4488, 0.4371, 0.4040)).view(1,3,1,1)
        self.gaussian_pyramid = GaussianPyramid(opt)
        
        for i in range(opt.n_cnn_trans):
            setattr(self, f'cnn_trans_{i}', CNNTransformation(opt, i))
        self.atb_trans_modules = []
        for i in range(opt.n_atb_trans):
            setattr(self, f'atb_trans_{i}', ATBTransformation(opt, i))
        
        self.to_out = nn.Conv2d(opt.cnn_dims[-1], 3, kernel_size=1)
    
    def forward(self, image, P1, P2):
        '''
        Input:
            image: (B,3,H,W)
            P1: (B,18,H/4,W/4)
            P2: (B,18,H/4,W/4)
        Output:
            image: (B,3,H,W)
        '''
        x = image-self.mean.to(image.device)
        x = torch.cat([x,P1,P2], dim=1)
        pyr = self.gaussian_pyramid.pyramid_decom(x)
        pyr.reverse()
        for i in range(self.n_atb_trans):
            layer = getattr(self, f'atb_trans_{i}')
            if i==0:
                x = layer(pyr[i])
            else:
                x = layer(torch.cat([x, pyr[i]], dim=1))
            x = self.gaussian_pyramid.upsample(x)
        
        for i in range(self.n_cnn_trans):
            layer = getattr(self, f'cnn_trans_{i}')
            x = layer(torch.cat([x, pyr[i+self.n_atb_trans]], dim=1))
            if i!=self.n_cnn_trans-1:
                x = self.gaussian_pyramid.upsample(x)
        
        x = self.to_out(x)
        return x