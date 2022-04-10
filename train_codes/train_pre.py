import argparse
import datetime
import json
import os
import random
import shutil
import time

import lpips
import numpy as np
import torch
import torch.utils
from easydict import EasyDict
from PIL import Image
from torch.nn import functional as F

from dataloader import DeepFashionTrainDataset, DeepFashionValDataset
from dataloader import Market1501TrainDataset, Market1501ValDataset
from losses import VGGLoss
from metrics import calculate_psnr, calculate_ssim
from model.pose_transformer import PoseTransformer
from utils.pose_utils import draw_pose_from_map
from utils.utils import load_option, tensor2ndarray, send_line_notify


def train(opt_path):
    opt = EasyDict(load_option(opt_path))
    
    random.seed(opt.pre.seed)
    np.random.seed(opt.pre.seed)
    torch.manual_seed(opt.pre.seed)
    if opt.pre.reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_name = opt.pre.name
    batch_size = opt.pre.batch_size
    print_freq = opt.pre.print_freq
    eval_freq = opt.pre.eval_freq
    
    model_ckpt_dir = f'./experiments/{model_name}/ckpt'
    image_out_dir = f'./experiments/{model_name}/generated'
    log_dir = f'./experiments/{model_name}/logs'
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/log_{model_name}.log', mode='w', encoding='utf-8') as fp:
        fp.write('')
    with open(f'{log_dir}/train_losses_{model_name}.csv', mode='w', encoding='utf-8') as fp:
        fp.write('step,lr_G,loss_G,l1loss,ploss,sloss\n')
    with open(f'{log_dir}/test_losses_{model_name}.csv', mode='w', encoding='utf-8') as fp:
        fp.write('step,loss_G,psnr,ssim,lpips\n')
    
    shutil.copy(opt_path, f'./experiments/{model_name}/{os.path.basename(opt_path)}')
    
    netG = PoseTransformer(opt).to(device)
    if opt.pre.pretrained_path:
        netG_state_dict = torch.load(opt.pre.pretrained_path, map_location=device)
        netG_state_dict = netG_state_dict['netG_state_dict']
        netG.load_state_dict(netG_state_dict, strict=False)

    perceptual_loss = VGGLoss().to(device)
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    loss_fn_alex.eval()
    
    optimG = torch.optim.Adam(netG.parameters(), lr=opt.pre.learning_rate_G, betas=opt.pre.betas)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=opt.pre.milestones, gamma=0.5)
    
    if opt.dataset_type=='fashion':
        train_dataset = DeepFashionTrainDataset(res=(256,256), pose_res=(256,256), dataset_path=opt.dataset_path)
        val_dataset = DeepFashionValDataset(res=(256,256), pose_res=(256,256), dataset_path=opt.dataset_path)
    elif opt.dataset_type=='market':
        train_dataset = Market1501TrainDataset(res=(128,64), pose_res=(128,64), dataset_path=opt.dataset_path)
        val_dataset = Market1501ValDataset(res=(128,64), pose_res=(128,64), dataset_path=opt.dataset_path)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print('Start Training')
    start_time = time.time()
    total_step = 0
    netG.train()
    for e in range(1, 9999):
        for i, data_dict in enumerate(train_loader):
            P1, P2, map1, map2, P1_path, P2_path = data_dict.values()
            P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
            b_size = P1.size(0)

            # Training G
            netG.zero_grad()
            fake = netG(P1,map1,map2)
            fake_img = fake.sigmoid()
            l1loss = opt.pre.coef_l1*F.l1_loss(fake_img, P2)
            ploss, sloss = perceptual_loss(fake_img, P2)
            ploss, sloss = opt.pre.coef_perc*ploss, opt.pre.coef_style*sloss
            loss_G = l1loss + ploss + sloss
            
            loss_G.backward()
            optimG.step()
            
            total_step += 1
            
            schedulerG.step()
            
            if total_step%1==0:
                lr_G = [group['lr'] for group in optimG.param_groups]
                with open(f'{log_dir}/train_losses_{model_name}.csv', mode='a', encoding='utf-8') as fp:
                    txt = f'{total_step},{lr_G[0]},{loss_G:f},{l1loss:f},{ploss:f},{sloss:f}\n'
                    fp.write(txt)
            
            if total_step%print_freq==0 or total_step==1:
                rest_step = opt.pre.steps-total_step
                time_per_step = int(time.time()-start_time) / total_step

                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{opt.pre.steps}, Epoch:{e:03}, elepsed: {elapsed}, '
                lg = lg + f'eta: {eta}, loss_G: {loss_G:f}, l1loss: {l1loss:f}, '
                lg = lg + f'ploss: {ploss:f}, sloss: {sloss:f}'
                print(lg)
                with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
            
            if total_step%eval_freq==0:
                # Validation
                netG.eval()
                val_step = 1
                psnr_fake = 0.0
                ssim_fake = 0.0
                lpips_val = 0.0
                loss_G_val = 0.0
                for j, val_data_dict in enumerate(val_loader):
                    P1, P2, map1, map2, P1_path, P2_path = val_data_dict.values()
                    P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
                    with torch.no_grad():
                        fake_val_logits = netG(P1,map1,map2)
                        fake_vals = fake_val_logits.sigmoid()
                        
                        l1loss = opt.pre.coef_l1*F.l1_loss(fake_vals, P2)
                        ploss, sloss = perceptual_loss(fake_vals, P2)
                        ploss, sloss = opt.pre.coef_perc*ploss, opt.pre.coef_style*sloss
                        loss_G_val += (l1loss + ploss + sloss) / (opt.pre.val_step/batch_size)
                    
                    lpips_val += loss_fn_alex(fake_vals, P2, normalize=True).sum()
                    
                    input_vals = tensor2ndarray(P1)
                    fake_vals = tensor2ndarray(fake_vals)
                    real_vals = tensor2ndarray(P2)
                    
                    for b in range(fake_vals.shape[0]):
                        if total_step%eval_freq==0:
                            os.makedirs(os.path.join(image_out_dir, f'{total_step:06}'), exist_ok=True)
                            psnr_fake += calculate_psnr(
                                fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=4, test_y_channel=True)
                            ssim_fake += calculate_ssim(
                                fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=4, test_y_channel=True)
                                            
                            # Visualization
                            mp1 = map1[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
                            mp1, _ = draw_pose_from_map(mp1)
                            mp2 = map2[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
                            mp2, _ = draw_pose_from_map(mp2)

                            input_val = Image.fromarray(input_vals[b,:,:,:])
                            mp1 = Image.fromarray(mp1)
                            fake_val = Image.fromarray(fake_vals[b,:,:,:])
                            mp2 = Image.fromarray(mp2)
                            real_val = Image.fromarray(real_vals[b,:,:,:])
                            img = Image.new('RGB', size=(5*input_val.width, input_val.height), color=0)
                            img.paste(input_val, box=(0, 0))
                            img.paste(mp1, box=(input_val.width, 0))
                            img.paste(fake_val, box=(2*input_val.width, 0))
                            img.paste(mp2, box=(3*input_val.width, 0))
                            img.paste(real_val, box=(4*input_val.width, 0))
                            img.save(os.path.join(image_out_dir, f'{total_step:06}', f'{j:03}_{b:02}.jpg'), 'JPEG')

                        val_step += 1
                        if val_step==opt.pre.val_step: break
                    if val_step==opt.pre.val_step: break

                psnr_fake = psnr_fake / val_step
                ssim_fake = ssim_fake / val_step
                lpips_val = lpips_val / val_step
                
                txt = f'PSNR: {psnr_fake:f}, SSIM: {ssim_fake:f}, LPIPS: {lpips_val:f}, loss_G: {loss_G_val:f}'
                print(txt)
                with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                    fp.write(txt+'\n')
                with open(f'{log_dir}/test_losses_{model_name}.csv', mode='a', encoding='utf-8') as fp:
                    fp.write(f'{total_step},{loss_G_val:f},{psnr_fake:f},{ssim_fake:f},{lpips_val:f}\n')
                
                if total_step%(50*eval_freq)==0 and opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'{opt.pre.name} Step: {total_step}\n{lg}\n{txt}')

                torch.save({
                    'total_step': total_step,
                    'netG_state_dict': netG.state_dict(),
                    'optimG_state_dict': optimG.state_dict(),
                    'PSNR': psnr_fake, 
                    'SSIM': ssim_fake
                }, os.path.join(model_ckpt_dir, f'{model_name}_{total_step:06}.ckpt'))
            
            if total_step==opt.pre.steps:
                if opt.enable_line_nortify:
                    with open('line_nortify_token.json', 'r', encoding='utf-8') as fp:
                        token = json.load(fp)['token']
                    send_line_notify(token, f'Complete training {opt.pre.name}.')
                
                print('Completed.')
                exit()
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of training network without adversarial loss.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    args = parser.parse_args()
    
    train(args.config)