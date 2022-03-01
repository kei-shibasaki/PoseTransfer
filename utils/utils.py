import json
import os
import numpy as np
import torch

def load_option(opt_path):
    with open(opt_path, 'r') as json_file:
        json_obj = json.load(json_file)
        return json_obj

def tensor2ndarray(tensor):
    # Pytorch Tensor (B, C, H, W), [0, 1] -> ndarray (B, H, W, C) [0, 255]
    img = tensor.detach()
    img = img.cpu().permute(0,2,3,1).numpy()
    img = np.clip(img, a_min=0, a_max=1.0)
    img = (img*255).astype(np.uint8)
    return img