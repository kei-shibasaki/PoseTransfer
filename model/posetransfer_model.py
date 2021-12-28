import torch 
from torch import nn
from model.transformer_block import SourcePoseEncoder, OuterDecoder, InnerDecoder
from model.CPE import ConditionalPositionalEncoding

class PoseTransferModel(nn.Module):
    def __init__(self, opt):
        opt_spe = opt['SourcePoseEncoder']
        opt_out = opt['OuterDecoder']
        opt_inn = opt['InnerDecoder']
        
        self.expand_dim_source = nn.Conv2d(3, opt_spe['dim'], kernel_size=1, bias=False)
        self.expand_dim_target = nn.Conv2d(3, opt_out['dim'], kernel_size=1, bias=False)
        
        self.source_pose_encoder = SourcePoseEncoder(
            opt_spe['dim'], opt_spe['heads'], opt_spe['mlp_ratio'], opt_spe['n_block'])
        self.outer_decoder = OuterDecoder(
            opt_out['dim'], opt_out['heads'], opt_out['mlp_ratio'], opt_out['n_block'])
        self.inner_decoder = InnerDecoder(
            opt_inn['dim'], opt_inn['heads'], opt_inn['mlp_ratio'], opt_inn['n_block'])
        
    def forward(self, P1, map1, P2, map2):
        P1 = self.expand_dim_source(P1)