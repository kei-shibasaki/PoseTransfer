import torch
from torch import nn
from torch.nn import functional as F

from model.axial_transformer import AxialTransformerBlock, ChanLayerNorm, AxialTransformerDecoderBlock
from model.positional_encoding import ConditionalPositionalEncoding

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize):
        super(ResBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize//2)
        self.norm = ChanLayerNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels!=out_channels:
            self.expand_dims = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.expand_dims = nn.Identity()
    
    def forward(self, x):
        a = self.expand_dims(x)
        x = self.cnn(x)
        x = self.norm(x)
        x = self.relu(x)
        x = a + x
        return x

class ShallowFeatureExtraction(nn.Module):
    def __init__(self, opt):
        super(ShallowFeatureExtraction, self).__init__()
        in_channels = opt.color_channels + opt.pose_channels
        dims = opt.dims
        self.fext_depths = opt.feature_extraction.depths
        fext_ksizes = opt.feature_extraction.ksizes
        
        for i, depth in enumerate(self.fext_depths):
            for j in range(depth):
                if i==j==0:
                    conv = nn.Conv2d(in_channels, dims[i], fext_ksizes[i], padding=fext_ksizes[i]//2)
                    norm = ChanLayerNorm(dims[i])
                    relu = nn.ReLU(inplace=True)
                elif j==0:
                    conv = nn.Conv2d(dims[i-1], dims[i], fext_ksizes[i], padding=fext_ksizes[i]//2)
                    norm = ChanLayerNorm(dims[i])
                    relu = nn.ReLU(inplace=True)
                else:
                    conv = nn.Conv2d(dims[i], dims[i], fext_ksizes[i], padding=fext_ksizes[i]//2)
                    norm = ChanLayerNorm(dims[i])
                    relu = nn.ReLU(inplace=True)
                setattr(self, f'conv2d_{i}_{j}', conv)
                setattr(self, f'layernorm_{i}_{j}', norm)
                setattr(self, f'relu_{i}_{j}', relu)
    
    def forward(self, img):
        img_pyr = []
        for i, depth in enumerate(self.fext_depths):
            for j in range(depth):
                conv = getattr(self, f'conv2d_{i}_{j}')
                lnorm = getattr(self, f'layernorm_{i}_{j}')
                relu = getattr(self, f'relu_{i}_{j}')
                
                img = relu(lnorm(conv(img)))
            img_pyr.append(img)
            
            img = F.avg_pool2d(img, kernel_size=2)
        
        img_pyr.append(img)
        return img_pyr

class PosePyramid(nn.Module):
    def __init__(self, opt):
        super(PosePyramid, self).__init__()
        self.level = opt.level
    
    def forward(self, pose):
        pose_pyr = [pose]
        for i in range(self.level):
            pose = F.max_pool2d(pose, kernel_size=2)
            pose_pyr.append(pose)
        return pose_pyr

class CNNTransformation(nn.Module):
    def __init__(self, opt, idx):
        super(CNNTransformation, self).__init__()
        depth = opt.cnn_trans.depths[idx]
        dim = opt.dims[-opt.n_at_trans-idx]
        ksize = opt.cnn_trans.ksizes[idx]
        cnn_in_channel = opt.dims[-opt.n_at_trans-idx+1]
        stat_dim = opt.dims[-opt.n_at_trans-idx]
        stat_ksize = opt.to_stat.ksizes[idx]
        
        self.to_stat = nn.Sequential(
            nn.Conv2d(stat_dim, dim, stat_ksize, padding=stat_ksize//2),
            ChanLayerNorm(dim), 
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, stat_ksize, padding=stat_ksize//2), 
            ChanLayerNorm(dim), 
            nn.ReLU(inplace=True))
        
        self.expand_dims = nn.Conv2d(cnn_in_channel, dim, kernel_size=1)
        layers = []
        for d in range(depth):
            if d==0:
                layers.append(ResBlock(dim*2, dim, ksize))
            else:
                layers.append(ResBlock(dim, dim, ksize))
        self.layers = nn.Sequential(*layers)
    
    def calc_mean_std(self, x):
        B, C, _, _ = x.shape

        mean = torch.mean(x.view(B, C, -1), 2).view(B, C, 1, 1)
        std = torch.std(x.view(B, C, -1), 2).view(B, C, 1, 1) + 1e-10

        return mean, std
    
    def adaptive_inistance_normalization(self, x, y):
        x_mean, x_std = self.calc_mean_std(x)
        y_mean, y_std = self.calc_mean_std(y)
        
        x_mean = x_mean.expand_as(x)
        x_std = x_std.expand_as(x)
        y_mean = y_mean.expand_as(x)
        y_std = y_std.expand_as(x)
        
        return y_std * (x - x_mean) / x_std + y_mean
    
    def forward(self, img, feature):
        y = self.to_stat(img)
        feature = self.expand_dims(feature)
        x = self.adaptive_inistance_normalization(feature, y)
        x = torch.cat([img, feature], dim=1)
        x = self.layers(x)
        return x

class Encoder(nn.Module):
    def __init__(self, opt, idx):
        super(Encoder, self).__init__()
        
        dim = opt.dims[min(-idx, -1)]
        heads = opt.at_trans.heads[idx]
        mlp_ratio = opt.at_trans.mlp_ratios[idx]
        drop = opt.at_trans.drops[idx]
        cpe_ksize = opt.at_trans.cpe_ksizes[idx]
        depth = opt.at_trans.encoder_depths[idx]
        
        if idx==0:
            in_channels = opt.dims[-1]
        elif idx==1:
            in_channels = opt.dims[-1] + opt.dims[-1]
        else:
            in_channels = opt.dims[-idx+1] + opt.dims[-idx]
        
        self.expand_dims = nn.Conv2d(in_channels, dim, kernel_size=1)
        self.cpe = ConditionalPositionalEncoding(dim, cpe_ksize)
        
        layers = []
        for i in range(depth):
            layers.append(AxialTransformerBlock(dim, heads=heads, mlp_ratio=mlp_ratio, dropout=drop))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.expand_dims(x)
        x = self.cpe(x)
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, opt, idx):
        super(Decoder, self).__init__()
        pose_channels = opt.pose_channels
        
        # idx: -1, -1, -2, -3 と選びたい
        dim = opt.dims[min(-idx, -1)]
        heads = opt.at_trans.heads[idx]
        mlp_ratio = opt.at_trans.mlp_ratios[idx]
        drop = opt.at_trans.drops[idx]
        cpe_ksize = opt.at_trans.cpe_ksizes[idx]
        depth = opt.at_trans.decoder_depths[idx]
        
        self.expand_dims = nn.Conv2d(pose_channels, dim, kernel_size=1)
        self.cpe = ConditionalPositionalEncoding(dim, cpe_ksize)
        
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(AxialTransformerDecoderBlock(dim, heads=heads, mlp_ratio=mlp_ratio, dropout=drop))
        
    def forward(self, pose, encoded):
        x = self.expand_dims(pose)
        x = self.cpe(x)
        for layer in self.layers:
            x = layer(x, encoded)
        return x

class AxialTransformerTransformation(nn.Module):
    def __init__(self, opt, idx):
        super(AxialTransformerTransformation, self).__init__()
        self.encoder = Encoder(opt, idx)
        self.decoder = Decoder(opt, idx)
    
    def forward(self, feature, pose):
        # feature: (B,C,H,W), pose: (B,18+18,H,W)
        encoded = self.encoder(feature)
        out = self.decoder(pose, encoded)
        return out


### Main Model ###
class PoseTransformer(nn.Module):
    def __init__(self, opt):
        # opt class Config: See details in config/config.py
        super(PoseTransformer, self).__init__()
        self.n_cnn_trans = opt.n_cnn_trans
        self.n_at_trans = opt.n_at_trans
        
        self.mean = torch.tensor((0.4488, 0.4371, 0.4040)).view(1,3,1,1)
        self.shallow_fext = ShallowFeatureExtraction(opt)
        self.pose_pyr = PosePyramid(opt)
        
        for i in range(opt.n_cnn_trans):
            setattr(self, f'cnn_trans_{i}', CNNTransformation(opt, i))
        self.at_trans_modules = []
        for i in range(opt.n_at_trans):
            setattr(self, f'at_trans_{i}', AxialTransformerTransformation(opt, i))
        
        self.to_out = nn.Conv2d(opt.dims[0], 3, kernel_size=1)
    
    def forward(self, image, P1, P2):
        f_i = image-self.mean.to(image.device)
        f_i = torch.cat([f_i, P1], dim=1)
        img_pyr = self.shallow_fext(f_i)
        p2_pyr = self.pose_pyr(P2)
        img_pyr.reverse()
        p2_pyr.reverse()

        for i in range(self.n_at_trans):
            layer = getattr(self, f'at_trans_{i}')
            if i==0:
                feature = layer(img_pyr[i], p2_pyr[i])
            else:
                feature = torch.cat([img_pyr[i], feature], dim=1)
                feature = layer(feature, p2_pyr[i])
            _, _, h, w = img_pyr[i+1].shape
            # avoiding error in concatenating feature with odd size
            feature = F.interpolate(feature, size=(h,w), mode='bilinear', align_corners=False)
        
        for i in range(self.n_cnn_trans):
            layer = getattr(self, f'cnn_trans_{i}')
            feature = layer(img_pyr[i+self.n_at_trans], feature)
            if i!=self.n_cnn_trans-1:
                _, _, h, w = img_pyr[i+self.n_at_trans+1].shape
                # avoiding error in concatenating feature with odd size
                feature = F.interpolate(feature, size=(h,w), mode='bilinear', align_corners=False)
        
        logits = self.to_out(feature)
        return logits