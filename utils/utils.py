import json
import os
import numpy as np
import torch
import requests
import pandas as pd

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

def get_best_checkpoint(experiment_dir, metrics):
    assert metrics in ['loss_G', 'PSNR', 'SSIM', 'LPIPS']
    model_name = os.path.basename(experiment_dir)
    ckpt_dir = os.path.join(experiment_dir, 'ckpt')
    log_dir = os.path.join(experiment_dir, 'logs')
    test_log = pd.read_csv(os.path.join(log_dir, f'test_losses_{model_name}.csv'))
    
    if metrics=='loss_G':
        best_step = test_log.loc[test_log['loss_G'].idxmin(), 'step']
    elif metrics=='PSNR':
        best_step = test_log.loc[test_log['psnr'].idxmax(), 'step']
    elif metrics=='SSIM':
        best_step = test_log.loc[test_log['ssim'].idxmax(), 'step']
    else:
        best_step = test_log.loc[test_log['lpips'].idxmin(), 'step']
    
    ckpt_path = os.path.join(ckpt_dir, f'{model_name}_{best_step:06}.ckpt')
    
    return ckpt_path