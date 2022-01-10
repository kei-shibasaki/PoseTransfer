import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import collections

from model.attention_block import ChanLayerNorm
from model.transformer_block import SourcePoseEncoder, OuterDecoder, InnerDecoder
from model.CPE import ConditionalPositionalEncoding
from utils import convert_bits, labels_to_bins, bins_to_labels

class PoseTransferModel(nn.Module):
    def __init__(self, opt):
        super(PoseTransferModel, self).__init__()
        self.opt = opt
        opt_spe = opt['SourcePoseEncoder']
        opt_out = opt['OuterDecoder']
        opt_inn = opt['InnerDecoder']
        self.num_symbols_per_channel = 2**3
        self.num_symbols = self.num_symbols_per_channel**3
        
        #rgb_mean = (0.4488, 0.4371, 0.4040)
        #self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        
        self.expand_dim_source = nn.Conv2d(3+18+18, opt['hidden_dim'], kernel_size=1, bias=False)
        self.expand_dim_target = nn.Conv2d(self.num_symbols, opt['hidden_dim'], kernel_size=1, bias=False)
        self.cpe_source = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        self.cpe_target = ConditionalPositionalEncoding(opt['hidden_dim'], opt['CPEk'])
        
        self.source_pose_encoder = SourcePoseEncoder(
            opt['hidden_dim'], opt['heads'], opt['mlp_ratio'], opt_spe['n_block'])
        self.outer_decoder = OuterDecoder(
            opt['hidden_dim'], opt['heads'], opt['mlp_ratio'], opt_out['n_block'])
        self.inner_decoder = InnerDecoder(
            opt['hidden_dim'], opt['heads'], opt['mlp_ratio'], opt_inn['n_block'])
        
        self.to_out = nn.Sequential(
            ChanLayerNorm(opt['hidden_dim']),
            nn.Conv2d(opt['hidden_dim'], 3, kernel_size=1, bias=False))
        
        self.pixel_embed_layer = nn.Linear(self.num_symbols, opt['hidden_dim'], bias=False)
        
    def forward(self, P1, map1, P2, map2):
        z = self.encoder(P1, map1, map2)
        dec_logits = self.decoder(P2, z)

        return dec_logits
    
    def encoder(self, P1, map1, map2):
        z = self.expand_dim_source(torch.cat([P1,map1,map2], dim=1))
        z = self.cpe_source(z)
        z = self.source_pose_encoder(z)
        return z
    
    def decoder(self, P2, z):
        # [0,1]->[0,255]
        labels = (P2*255).to(torch.int32)
        labels = convert_bits(labels, n_bits_in=8, n_bits_out=3)
        labels = labels_to_bins(labels, self.num_symbols_per_channel)
        # (B,H,W)->(B,H,W,512)->(B,512,H,W)
        labels = F.one_hot(labels, num_classes=self.num_symbols).permute(0,3,1,2).float()
        
        h_dec = self.expand_dim_target(labels)
        h_upper = self.outer_decoder(h_dec, z)
        h_inner = self.inner_decoder(h_dec, h_upper, z)
        
        activations = self.to_out(h_inner)
        return activations
    
    def image_loss(self, images, targets):
        loss = F.l1_loss(images, targets)
        return loss
    
    def loss(self, logits, targets):
        images = torch.tanh(logits)
        loss = self.image_loss(images, targets)
        return loss
    
    def autoregressive_sample(self, z, mode='sample'):
        """Generates pixel-by-pixel.
        1. The encoder is run once per-channel.
        2. The outer decoder is run once per-row.
        3. the inner decoder is run once per-pixel.
        The context from the encoder and outer decoder conditions the
        inner decoder. The inner decoder then generates a row, one pixel at a time.
        After generating all pixels in a row, the outer decoder is run to recompute
        context. This condtions the inner decoder, which then generates the next
        row, pixel-by-pixel.
        Args:
        z_gray: grayscale image.
        mode: sample or argmax.
        Returns:
        image: coarse image of shape (B, H, W)
        image_proba: probalities, shape (B, H, W, 512)
        """
        b, c, h, w = z.shape
        # channel_cache[i, j] stores the pixel embedding for row i and col j.
        channel_cache = torch.zeros_like(z).to(z.device)
        # upper_context[row_ind] stores context from all previously generated rows.
        upper_context = torch.zeros_like(z).to(z.device)
        # row_cache[0, j] stores the pixel embedding for the column j of the row
        # under generation. After every row is generated, this is rewritten.
        row_cache = torch.zeros(size=(b, c, 1, w)).to(z.device)
        
        pixel_samples, pixel_probas = [], []
        with torch.no_grad():
            for row in range(h):
                # (B,C,H,W)->(B,C,W)->(B,C,1,W)
                row_cond_channel = z[:, :, row, :].unsqueeze(2)
                # (B,C,H,W)->(B,C,W)->(B,C,1,W)
                row_cond_upper = upper_context[:, :, row, :].unsqueeze(2)
                
                gen_row, proba_row = [], []
                
                for col in range(w):
                    # (B,C,1,W)->(B,C,1,W)
                    activations = self.inner_decoder(row_cache, row_cond_upper, row_cond_channel)
                    # pixel_sample: (B,), pixel_embed: (B,C,1,1), pixel_proba: (B, num_symbols)
                    pixel_sample, pixel_embed, pixel_proba = self.act_logit_sample_embed(activations, col, mode=mode)
                    proba_row.append(pixel_proba)
                    gen_row.append(pixel_sample)
                    
                    row_cache[:, :, 0, col] = pixel_embed
                    channel_cache[:, :, row, col] = pixel_embed
                
                # [(B,)] -> (B,W)
                gen_row = torch.stack(gen_row, dim=-1)
                pixel_samples.append(gen_row)
                # [(B, num_symbols)]->(B,num_symbols,W)
                pixel_probas.append(torch.stack(proba_row, dim=-1))
                upper_context = self.outer_decoder(channel_cache, z)
            
            # [(B,W)]->(B,H,W)
            image = torch.stack(pixel_samples, dim=1)
            # (B,H,W)->(B,3,H,W)
            image = self.post_process_image(image)
            
            # [(B,num_symbols,W)]->(B,num_symbols,H,W)
            image_proba = torch.stack(pixel_probas, dim=2)
        
        return image, image_proba
       
    
    def act_logit_sample_embed(self, activations, col_ind, mode='sample'):
        """Converts activations[col_ind] to the output pixel.
        Activation -> Logit -> Sample -> Embedding.
        Args:
        activations: 5-D Tensor, shape=(batch_size, hidden_size, 1, width)
        col_ind: integer.
        mode: 'sample' or 'argmax'
        Returns:
        pixel_sample: 1-D Tensor, shape=(batch_size, )
        pixel_embed: 4-D Tensor, shape=(batch_size, hidden_size)
        pixel_proba: 4-D Tensor, shape=(batch_size, 512)
        """
        b_size = activations.size(0)
        # (B,C,1,W)->(B,C,1)->(B,C,1,1)
        pixel_activation = activations[:, :, :, col_ind].unsqueeze(3)
        # (B,C,1,1)->(B,num_symbols,1,1)->(B,num_symbols)
        pixel_logits = self.to_out(pixel_activation).squeeze(-1).squeeze(-1)
        pixel_proba = pixel_logits.softmax(dim=1)
        
        if mode=='sample':
            # (B,num_symbols)->(B,1)->(B,)
            pixel_sample = torch.multinomial(pixel_proba, num_samples=1).squeeze(1)
        elif mode=='argmax':
            # (B,num_symbols)->(B,)
            pixel_sample = torch.argmax(pixel_proba, dim=1)
        
        # (B,)->(B,num_symbol)
        pixel_one_hot = F.one_hot(pixel_sample, num_classes=self.num_symbols).float()
        pixel_embed = self.pixel_embed_layer(pixel_one_hot)
        
        return pixel_sample, pixel_embed, pixel_proba
    
    def post_process_image(self, image):
        image = bins_to_labels(image, self.num_symbols_per_channel)
        image = convert_bits(image, n_bits_in=3, n_bits_out=8)
        image = image
        return image