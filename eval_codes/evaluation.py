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

from utils.utils import tensor2ndarray, load_option
from utils.pose_utils import draw_pose_from_map
from dataloader import DeepFashionValDataset, Market1501ValDataset
from model.pose_transformer import PoseTransformer
from metrics import calculate_psnr, calculate_ssim
from skimage.metrics import structural_similarity
import lpips

def eval_from_image(opt_path, gen_path, gt_path):
    images_gen = sorted(glob.glob(os.path.join(gen_path, '*.jpg')))
    images_gt = sorted(glob.glob(os.path.join(gt_path, '*.jpg')))
    n_images = len(images_gen)
    
    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex.eval()
    
    psnr, ssim, lpips_val = 0.0, 0.0, 0.0
    for imgpath_gen, imgpath_gt in zip(tqdm(images_gen), images_gt):
        with Image.open(imgpath_gen) as img_gen, Image.open(imgpath_gt) as img_gt:
            img_gen = np.array(img_gen)
            img_gt = np.array(img_gt)
            psnr += calculate_psnr(img_gen, img_gt, crop_border=0, test_y_channel=True) / n_images
            ssim += calculate_ssim(img_gen, img_gt, crop_border=0, test_y_channel=True) / n_images
            img_gen = torch.tensor(img_gen).permute(2,0,1) / 255.0
            img_gt = torch.tensor(img_gt).permute(2,0,1) / 255.0
            lpips_val += loss_fn_alex(img_gen, img_gt, normalize=True) / n_images

    print(f'PSNR: {psnr:f}, SSIM: {ssim:f}, LPIPS: {lpips_val.detach().numpy()}')

def evaluation(batch_size, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = EasyDict(load_option(opt_path))
    print('Preparing Data...')
    if opt.dataset_type=='fashion':
        val_dataset = DeepFashionValDataset(res=(256,256), pose_res=(256,256), dataset_path=opt.dataset_path)
    elif opt.dataset_type=='market':
        val_dataset = Market1501ValDataset(res=(128,64), pose_res=(128,64), dataset_path=opt.dataset_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print('Creating Net...')
    net = PoseTransformer(opt).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['netG_state_dict'])
    loss_fn_alex = lpips.LPIPS(net='alex')
    
    print('Calculating...')
    net.eval()
    psnr_fake, ssim_fake, lpips_val = 0.0, 0.0, 0.0
    val_steps = len(val_dataset)
    for i, data_dict in enumerate(tqdm(val_loader)):
        P1, P2, map1, map2, P1_path, P2_path = data_dict.values()
        P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
        b_size, _, _, _ = P1.shape
        
        with torch.no_grad():
            out = net(P1,map1,map2)
            out = out.sigmoid()
        
        for b in range(fake_vals.shape[0]):
            lpips_val += loss_fn_alex(fake_vals[b,:,:,:], real_vals[b,:,:,:], normalize=True)
        
        input_vals = tensor2ndarray(P1)
        fake_vals = tensor2ndarray(out)
        real_vals = tensor2ndarray(P2)
        
        for b in range(fake_vals.shape[0]):
            psnr_fake += calculate_psnr(
                fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=0, test_y_channel=False)
            ssim_fake += calculate_ssim(
                fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=0, test_y_channel=False)
        
        psnr_fake = psnr_fake / val_steps
        ssim_fake = ssim_fake / val_steps
    
    print(f'PSNR: {psnr_fake:f}, SSIM: {ssim_fake:f}')
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of evaluate metrics.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    
    model_name = args.fine.name
    
    gen_path = f'results/{model_name}/generated'
    gt_path = f'results/{model_name}/GT'

    eval_from_image(args.config, gen_path, gt_path)