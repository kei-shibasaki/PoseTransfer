from statistics import mode
import torch
import torch.utils
from torch import nn
from torch.nn import functional as F
import os
import time
from PIL import Image
import datetime
import shutil

from model.pose_transformer import PoseTransformer
from model.discriminator import Discriminator
from dataloader import DeepFashionTrainDataset, DeepFashionValDataset
from utils.utils import tensor2ndarray
from utils.pose_utils import draw_pose_from_map, press_pose
from metrics import calculate_psnr, calculate_ssim
from losses import GANLoss, gradient_penalty, VGGPerceptualLoss
from config import config as cfg

def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    opt = cfg.Config()
    model_name = opt.name
    batch_size = opt.batch_size
    epoch = opt.epoch
    print_freq = opt.print_freq
    eval_freq = opt.eval_freq
    
    model_ckpt_dir = f'./experiments/{model_name}/ckpt'
    image_out_dir = f'./experiments/{model_name}/generated'
    log_dir = f'./experiments/{model_name}/logs'
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/log_{model_name}.log', mode='w', encoding='utf-8') as fp:
        fp.write(f'epoch: {epoch}, batch_size: {batch_size}, model_name: {model_name}\n')
    with open(f'{log_dir}/losses_{model_name}.csv', mode='w', encoding='utf-8') as fp:
        fp.write('step,lr_g,loss_g,lr_d,loss_d,loss_d_real,loss_d_fake\n')
    
    shutil.copy('./config/config.py', f'./experiments/{model_name}/config.py')
    
    netG = PoseTransformer(opt).to(device)
    netG_state_dict = torch.load(opt.pretrained_path, map_location=device)
    netG_state_dict = netG_state_dict['netG_state_dict']
    netG.load_state_dict(netG_state_dict, strict=False)
    perceptual_loss = VGGPerceptualLoss(resize=True).to(device)
    
    pretrained_names = ['crossing_swin_transformer', 'fusion_swin_transformer']
    
    pretrained_params = []
    other_params = []
    for name, param in netG.named_parameters():
        is_pretrained_param = False
        for pretrained_name in pretrained_names:
            is_pretrained_param = is_pretrained_param or (pretrained_name in name)
        if is_pretrained_param:
            param.requires_grad = True
            pretrained_params.append(param)
            # print(f'pretrained_params <- {name}')
        else:
            other_params.append(param)
            # print(f'other_params <- {name}')
    
    # optimG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5,0.999))
    optimG = torch.optim.Adam([
        {'params': pretrained_params, 'lr': 1e-5, 'betas': (0.5,0.999)},
        {'params': other_params, 'lr': 1e-4, 'betas': (0.5,0.999)}
        ])
    milestones = opt.milestones
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=milestones, gamma=0.5)
    
    train_dataset = DeepFashionTrainDataset(res=256, pose_res=64)
    val_dataset = DeepFashionValDataset(res=256, pose_res=64)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print('Start Training')
    start_time = time.time()
    total_step = 0
    netG.train()
    for e in range(1, epoch+1):
        for i, data_dict in enumerate(train_loader):
            P1, P2, map1, map2, P1_path, P2_path = data_dict.values()
            P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
            b_size = P1.size(0)

            # Training G
            netG.zero_grad()
            fake = netG(P1,map1,map2)
            fake_img = fake.sigmoid()
            l1_loss = F.l1_loss(fake_img, P2)
            ploss = perceptual_loss(fake_img, P2)
            loss_G = l1_loss + ploss
            
            loss_G.backward()
            optimG.step()
            
            total_step += 1
            
            schedulerG.step()
            
            if total_step%1==0:
                lr_G = [group['lr'] for group in optimG.param_groups]
                with open(f'{log_dir}/losses_{model_name}.csv', mode='a', encoding='utf-8') as fp:
                    txt = f'{total_step},{lr_G[0]},{loss_G.detach().cpu().numpy()}\n'
                    fp.write(txt)
            
            if total_step%print_freq==0 or total_step==1:
                rest_step = epoch*len(train_loader)-total_step
                time_per_step = int(time.time()-start_time) / total_step

                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{len(train_loader)*epoch}, Epoch:{e:03}, elepsed: {elapsed}, '
                lg = lg + f'eta: {eta}, loss_G: {loss_G:f}'
                print(lg)
                with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
            
            if total_step%eval_freq==0 or total_step%len(train_loader)==0:
                # Validation
                netG.eval()
                val_step = 1
                psnr_fake = 0.0
                ssim_fake = 0.0
                for j, val_data_dict in enumerate(val_loader):
                    P1, P2, map1, map2, P1_path, P2_path = val_data_dict.values()
                    P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
                    with torch.no_grad():
                        fake_val_logits = netG(P1,map1,map2)
                        fake_vals = fake_val_logits.sigmoid()
                    
                        # (64,64)->(256,256)
                        map1 = F.interpolate(map1, scale_factor=4, mode='bicubic', align_corners=False)
                        map2 = F.interpolate(map2, scale_factor=4, mode='bicubic', align_corners=False)
                    
                    input_vals = tensor2ndarray(P1)
                    fake_vals = tensor2ndarray(fake_vals)
                    real_vals = tensor2ndarray(P2)
                    
                    for b in range(fake_vals.shape[0]):
                        if total_step%eval_freq==0:
                            os.makedirs(os.path.join(image_out_dir, f'{total_step:06}'), exist_ok=True)
                            psnr_fake += calculate_psnr(
                                fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=4, test_y_channel=False)
                            ssim_fake += calculate_ssim(
                                fake_vals[b,:,:,:], real_vals[b,:,:,:], crop_border=4, test_y_channel=False)
                            
                            mp1 = map1[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
                            # mp1, _ = draw_pose_from_map(mp1)
                            mp1 = press_pose(mp1)
                            mp2 = map2[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
                            # mp2, _ = draw_pose_from_map(mp2)
                            mp2 = press_pose(mp2)

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
                        if val_step==opt.val_step: break
                    if val_step==opt.val_step: break

                psnr_fake = psnr_fake / val_step
                ssim_fake = ssim_fake / val_step
                
                if total_step%eval_freq==0:
                    txt = f'PSNR: {psnr_fake:f}, SSIM: {ssim_fake:f}'
                    print(txt)
                    with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                        fp.write(txt+'\n')

                    torch.save({
                        'total_step': total_step,
                        'netG_state_dict': netG.state_dict(),
                        'optimG_state_dict': optimG.state_dict(),
                        'PSNR': psnr_fake, 
                        'SSIM': ssim_fake
                    }, os.path.join(model_ckpt_dir, f'{model_name}_{total_step:06}.ckpt'))
        

if __name__=='__main__':
    torch.backends.cudnn.benchmark = True
    train()