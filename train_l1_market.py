import torch
import torch.utils
from torch import nn
from torch.nn import functional as F
import os
import time
from PIL import Image
import datetime
import shutil
from easydict import EasyDict

from model.pose_transformer_adain import PoseTransformer
from model.discriminator import Discriminator
from dataloader import DeepFashionTrainDataset, DeepFashionValDataset
from dataloader import Market1501TrainDataset, Market1501ValDataset
from utils.utils import tensor2ndarray, load_option
from utils.pose_utils import draw_pose_from_map, press_pose
from metrics import calculate_psnr, calculate_ssim
from losses import GANLoss, gradient_penalty, VGGLoss

def train(opt_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    opt = EasyDict(load_option(opt_path))
    model_name = opt.name
    batch_size = opt.batch_size
    print_freq = opt.print_freq
    eval_freq = opt.eval_freq
    
    model_ckpt_dir = f'./experiments/{model_name}/ckpt'
    image_out_dir = f'./experiments/{model_name}/generated'
    log_dir = f'./experiments/{model_name}/logs'
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/log_{model_name}.log', mode='w', encoding='utf-8') as fp:
        fp.write('')
    with open(f'{log_dir}/losses_{model_name}.csv', mode='w', encoding='utf-8') as fp:
        fp.write('')
    
    shutil.copy(opt_path, f'./experiments/{model_name}/{os.path.basename(opt_path)}')
    
    netG = PoseTransformer(opt).to(device)
    perceptual_loss = VGGLoss().to(device)
    
    optimG = torch.optim.Adam(netG.parameters(), lr=opt.learning_rate_G, betas=opt.betas)
    milestones = opt.milestones
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=milestones, gamma=0.5)
    
    train_dataset = Market1501TrainDataset(res=(128, 64), pose_res=(128, 64))
    val_dataset = Market1501ValDataset(res=(128, 64), pose_res=(128, 64))
    
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
            l1_loss = opt.coef_l1*F.l1_loss(fake_img, P2)
            ploss, sloss = perceptual_loss(fake_img, P2)
            ploss, sloss = opt.coef_perc*ploss, opt.coef_style*sloss
            loss_G = l1_loss + ploss + sloss
            
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
                rest_step = opt.steps-total_step
                time_per_step = int(time.time()-start_time) / total_step

                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{opt.steps}, Epoch:{e:03}, elepsed: {elapsed}, '
                lg = lg + f'eta: {eta}, loss_G: {loss_G:f}, l1_loss: {l1_loss:f}, '
                lg = lg + f'ploss: {ploss:f}, sloss: {sloss:f}'
                print(lg)
                with open(f'{log_dir}/log_{model_name}.log', mode='a', encoding='utf-8') as fp:
                    fp.write(lg+'\n')
            
            if total_step%eval_freq==0 or total_step%len(train_loader)==0:
                # Validation
                netG.eval()
                val_step = 1
                psnr_fake = 0.0
                ssim_fake = 0.0
                loss_G_val = 0.0
                for j, val_data_dict in enumerate(val_loader):
                    P1, P2, map1, map2, P1_path, P2_path = val_data_dict.values()
                    P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
                    with torch.no_grad():
                        fake_val_logits = netG(P1,map1,map2)
                        fake_vals = fake_val_logits.sigmoid()
                        
                        l1_loss = opt.coef_l1*F.l1_loss(fake_vals, P2)
                        ploss, sloss = perceptual_loss(fake_vals, P2)
                        ploss, sloss = opt.coef_perc*ploss, opt.coef_style*sloss
                        loss_G_val += (l1_loss + ploss + sloss) / (opt.val_step/opt.batch_size)
                    
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
                            mp1, _ = draw_pose_from_map(mp1)
                            # mp1 = press_pose(mp1)
                            mp2 = map2[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
                            mp2, _ = draw_pose_from_map(mp2)
                            # mp2 = press_pose(mp2)

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
                    txt = f'PSNR: {psnr_fake:f}, SSIM: {ssim_fake:f}, loss_G: {loss_G_val:f}'
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
            
            if total_step==opt.steps:
                print('Completed.')
                exit()
        

if __name__=='__main__':
    torch.backends.cudnn.benchmark = True
    opt_path = 'config/config_market.json'
    train(opt_path)