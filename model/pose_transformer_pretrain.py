import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from model.axial_transformer import AxialTransformerBlock
from model.positional_encoding import ConditionalPositionalEncoding



class CrossingAxialTransformerBlock(nn.Module):
    def __init__(self, opt):
        '''
        Args:
            opt class Config.CrossingSwinTransformerConfig: See details in config/config.py
            shift_size int: shift value of shifted window this is not determined in option
            drop_path float: ratio of droppath
        '''
        super(CrossingAxialTransformerBlock, self).__init__()
        
        self.axial_block_img_cross = AxialTransformerBlock(
            opt.dim, heads=opt.num_heads, mlp_ratio=opt.mlp_ratio, dropout=opt.drop)
        self.axial_block_img = AxialTransformerBlock(
            opt.dim, heads=opt.num_heads, mlp_ratio=opt.mlp_ratio, dropout=opt.drop)
        self.axial_block_pose_cross = AxialTransformerBlock(
            opt.dim, heads=opt.num_heads, mlp_ratio=opt.mlp_ratio, dropout=opt.drop)
        self.axial_block_pose = AxialTransformerBlock(
            opt.dim, heads=opt.num_heads, mlp_ratio=opt.mlp_ratio, dropout=opt.drop)
    
    def forward(self, img_feature, pose_feature):
        '''
        Input
            img_feature: image feature (B, C, H, W)
            pose_feature: pose feature (B, C, H, W)
        Output
            img_feature: image feature (B, C, H, W)
            pose_feature: pose feature (B, C, H, W)
        '''
        
        f_i = self.axial_block_img_cross(img_feature, kv=pose_feature)
        f_i = self.axial_block_img(f_i)
        f_p = self.axial_block_pose_cross(pose_feature, kv=img_feature)
        f_p = self.axial_block_pose(f_p)
        
        return f_i, f_p

class CrossingAxialTransformer(nn.Module):
    def __init__(self, opt):
        super(CrossingAxialTransformer, self).__init__()
        '''
        Args:
            opt class Config.CrossingSwinTransformerConfig: See details in config/config.py
        '''
        self.expand_dim_pose = nn.Conv2d(18+18, opt.dim, kernel_size=1)
        self.cpe_pose = ConditionalPositionalEncoding(opt.dim, k=3)
        
        self.layers = nn.ModuleList()
        for i in range(opt.depth):
            layer = CrossingAxialTransformerBlock(opt)
            self.layers.append(layer)
        
        # self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, img_feature, pose_feature):
        '''
        Input: 
            img_feature: (B,C,H,W)
            pose_feature: (B,18+18,H,W)
        Output: 
            img_feature: (B,C,H,W)
            pose_feature: (B,C,H,W)
        '''
        pose_feature = self.expand_dim_pose(pose_feature)
        pose_feature = self.cpe_pose(pose_feature)

        for layer in self.layers:
            img_feature, pose_feature = layer(img_feature, pose_feature)
        
        return img_feature, pose_feature

class FusionAxialTransformer(nn.Module):
    def __init__(self, opt):
        super(FusionAxialTransformer, self).__init__()
        '''
        Args:
            opt class Config.FusionSwinTransformerConfig: See details in config/config.py
        '''
        self.layers = nn.ModuleList()
        for i in range(opt.depth):
            layer = AxialTransformerBlock(
                opt.dim*2, heads=opt.num_heads, mlp_ratio=opt.mlp_ratio, dropout=opt.drop)
            self.layers.append(layer)

        self.to_out = nn.Conv2d(opt.dim*2, opt.dim, kernel_size=1)
    
    def forward(self, img_feature, pose_feature):
        '''
        Input: 
            img_feature: (B,C,H,W)
            pose_feature: (B,C,H,W)
        Output: 
            output_image: (B,C,H,W)
        '''
        x = torch.cat([img_feature, pose_feature], dim=1)
        
        for layer in self.layers:
            x = layer(x)
        
        out = self.to_out(x)
        
        return out

class ShallowFeatureExtraction(nn.Module):
    def __init__(self, opt, in_channels):
        '''
        Args:
            opt class Config.ShallowFeatureExtractionConfig: See details in config/config.py
        '''
        super(ShallowFeatureExtraction, self).__init__()
        
        self.layers1 = nn.ModuleList()
        for i in range(opt.depth1):
            if i==0:
                layer = nn.Conv2d(in_channels, opt.dim1, kernel_size=opt.ksize1, stride=1, padding=opt.ksize1//2)
            else:
                layer = nn.Conv2d(opt.dim1, opt.dim1, kernel_size=opt.ksize1, stride=1, padding=opt.ksize1//2)
            self.layers1.append(layer)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        
        self.layers2 = nn.ModuleList()
        for i in range(opt.depth2):
            if i==0:
                layer = nn.Conv2d(opt.dim1, opt.dim2, kernel_size=opt.ksize2, stride=1, padding=opt.ksize2//2)
            else:
                layer = nn.Conv2d(opt.dim2, opt.dim2, kernel_size=opt.ksize2, stride=1, padding=opt.ksize2//2)
            self.layers2.append(layer)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
    
    def forward(self, x):
        '''
        Input:
            x: (B,C,H,W)
        Output:
            f0: (B,opt.dim1,H,W)
            f1: (B,opt.dim2,H/2,W/2)
            x: (B,opt.dim2,H/4,W/4)
        '''
        for layer in self.layers1:
            x = layer(x)
        f0 = x
        x = self.avgpool1(x)

        for layer in self.layers2:
            x = layer(x)
        f1 = x
        x = self.avgpool2(x)
        
        return f0, f1, x
        
class ImageReconstruction(nn.Module):
    def __init__(self, opt, x_dim, f0_dim, f1_dim):
        '''
        Args:
            opt class Config.ImageReconstructionConfig: See details in config/config.py
        '''
        super(ImageReconstruction, self).__init__()
        
        self.layers1 = nn.ModuleList()
        for i in range(opt.depth1):
            if i==0:
                layer = nn.Conv2d(x_dim+f1_dim, opt.dim1, kernel_size=opt.ksize1, stride=1, padding=opt.ksize1//2)
            else:
                layer = nn.Conv2d(opt.dim1, opt.dim1, kernel_size=opt.ksize1, stride=1, padding=opt.ksize1//2)
            self.layers1.append(layer)
        
        self.layers2 = nn.ModuleList()
        for i in range(opt.depth2):
            if i==0:
                layer = nn.Conv2d(opt.dim1+f0_dim, opt.dim2, kernel_size=opt.ksize2, stride=1, padding=opt.ksize2//2)
            else:
                layer = nn.Conv2d(opt.dim2, opt.dim2, kernel_size=opt.ksize2, stride=1, padding=opt.ksize2//2)
            self.layers2.append(layer)
        
        self.to_out = nn.Conv2d(opt.dim2, 3, kernel_size=1)
    
    def forward(self, x, f0, f1):
        '''
        Input:
            x: output from FusionSwinTransformerBlock (B,C,H/4,W/4)
            f0: f0 from ShallowFeatureExtraction (B,C,H,W)
            f1: f1 from ShallowFeatureExtraction (B,C,H/2,W/2)
        Output:
            out: output image (B,3,H,W)
        '''
        x = F.interpolate(x, size=(f1.shape[2], f1.shape[3]), 
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, f1], dim=1)
        for layer in self.layers1:
            x = layer(x)
        
        x = F.interpolate(x, size=(f0.shape[2], f0.shape[3]), 
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, f0], dim=1)
        for layer in self.layers2:
            x = layer(x)
        
        out = self.to_out(x)
        return out

### Main Model ###
class PoseTransformer(nn.Module):
    def __init__(self, opt):
        '''
        Args:
            opt class Config: See details in config/config.py
        '''
        super(PoseTransformer, self).__init__()
        self.mean = torch.tensor((0.4488, 0.4371, 0.4040)).view(1,3,1,1)
        opt_cross_swin = opt.CrossingSwinTransformerConfig()
        opt_fusion_swin = opt.FusionSwinTransformerConfig()
        
        self.expand_dim_img = nn.Conv2d(3, opt_cross_swin.dim, kernel_size=1)
        self.cpe_img = ConditionalPositionalEncoding(opt_cross_swin.dim)
        
        self.crossing_swin_transformer = CrossingAxialTransformer(opt_cross_swin)
        self.fusion_swin_transformer = FusionAxialTransformer(opt_fusion_swin)
        
        self.to_rgb = nn.Conv2d(opt_fusion_swin.dim, 3, kernel_size=1)
    
    def forward(self, image, P1, P2):
        '''
        Input:
            image: (B,3,H,W)
            P1: (B,18,H,W)
            P2: (B,18,H,W)
        Output:
            image: (B,3,H,W)
        '''
        x = image-self.mean.to(image.device)
        x = self.cpe_img(self.expand_dim_img(x))
        
        pose = torch.cat([P1, P2], dim=1)
        
        x, pose = self.crossing_swin_transformer(x, pose)
        x = self.fusion_swin_transformer(x, pose)
        
        
        rgb = self.to_rgb(x)
        
        return rgb