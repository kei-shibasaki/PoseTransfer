import glob
import os
import argparse

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms.functional
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import numpy as np
from easydict import EasyDict

from dataloader import SimpleImageDataset
from metrics import calculate_psnr, calculate_ssim, inception_score
from utils.utils import load_option
import lpips

def eval_from_image(out_path, gen_path, gt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images_gen = sorted(glob.glob(os.path.join(gen_path, '*.jpg')))
    images_gt = sorted(glob.glob(os.path.join(gt_path, '*.jpg')))
    n_images = len(images_gen)
    
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_alex.eval()
    
    psnr, ssim, lpips_val = 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgpath_gen, imgpath_gt in zip(tqdm(images_gen), images_gt):
            with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
                img_gen = np.array(img_gen)
                img_gt = np.array(img_gt)
                psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=True) / n_images
                ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=True) / n_images
                img_gen = (torch.tensor(img_gen).permute(2,0,1) / 255.0).to(device)
                img_gt = (torch.tensor(img_gt).permute(2,0,1) / 255.0).to(device)
                lpips_val += loss_fn_alex(img_gen, img_gt, normalize=True).sum() / n_images
    
        dataset_gen = SimpleImageDataset(gen_path)
<<<<<<< Updated upstream
        # dataset_gt = SimpleImageDataset(gt_path)
=======
>>>>>>> Stashed changes
        
        is_score_gen, _ = inception_score(dataset_gen, cuda=True, batch_size=16, resize=True, splits=10)

    print(f'PSNR: {psnr:f}, SSIM: {ssim:f}, LPIPS: {lpips_val:f}, IS: {is_score_gen:f}')
    
    with open(os.path.join(out_path, 'results.txt'), 'w', encoding='utf-8') as fp:
        fp.write(f'PSNR: {psnr:f}, SSIM: {ssim:f}, LPIPS: {lpips_val:f}, IS: {is_score_gen:f}')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of evaluate metrics.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
    model_name = opt.name
    
<<<<<<< Updated upstream
    if args.mode=='pre':
        model_name = opt.pre.name
    else:
        model_name = opt.fine.name
    
=======
>>>>>>> Stashed changes
    out_path = f'results/{model_name}'
    gen_path = f'results/{model_name}/generated'
    gt_path = f'results/{model_name}/GT'
    

    eval_from_image(out_path, gen_path, gt_path)