import json
import os
import numpy as np
import torch
import requests

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
    
def send_line_notify(line_notify_token, nortification_message):
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'{nortification_message}'}
    requests.post(line_notify_api, headers=headers, data=data)