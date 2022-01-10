import torch
from torch.nn.modules import loss
import torch.utils
from torch import nn
import os
import time
from PIL import Image
import datetime

from model_simple.posetransfer_model import SimplePoseTransferModel
from model_simple.discriminator import Discriminator
from dataloader import DeepFashionTrainDataset, DeepFashionValDataset
from utils import tensor2ndarray
from metrics import calculate_psnr, calculate_ssim
from losses import GANLoss, gradient_penalty

def train(epoch, batch_size, model_name, print_freq, eval_freq):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
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
    
    netG = SimplePoseTransferModel().to(device)
    netD = Discriminator().to(device)
    
    criterion = GANLoss(r=1e2, method='hinge')
    optimG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.9,0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9,0.999))
    milestonesG = [250000, 400000]
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=milestonesG, gamma=0.5)
    milestonesD = [250000, 400000]
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimG, milestones=milestonesD, gamma=0.5)
    
    train_dataset = DeepFashionTrainDataset()
    val_dataset = DeepFashionValDataset()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    print('Start Training')
    start_time = time.time()
    total_step = 0
    netG.train()
    for e in range(1, epoch+1):
        for i, data_dict in enumerate(train_loader):
            P1, P2, map1, map2, P1_path, P2_path = data_dict.values()
            P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
            b_size = P1.size(0)
            
            # Training D
            critic_iter = 1
            for c in range(critic_iter):
                logits_real = netD(P2)
                loss_D_real = criterion.discriminator_loss_real(logits_real)
                
                fake = netG(P1,map1,map2)
                logits_fake = netD(fake.detach())
                loss_D_fake = criterion.discriminator_loss_fake(logits_fake)
                
                gp = gradient_penalty(netD, P2, fake)
                
                loss_D = loss_D_real + loss_D_fake + 10*gp
                
                loss_D.backward()
                optimD.step()
            
            # Training G
            netG.zero_grad()
            logits_fake = netD(fake)
            loss_G = criterion.generator_loss(fake, P2, logits_fake)
            
            loss_G.backward()
            optimG.step()
            
            total_step += 1
            
            schedulerG.step()
            
            if total_step%1==0:
                lr_G = [group['lr'] for group in optimG.param_groups]
                lr_D = [group['lr'] for group in optimD.param_groups]
                with open(f'{log_dir}/losses_{model_name}.csv', mode='a', encoding='utf-8') as fp:
                    txt = f'{total_step},{lr_G[0]},{loss_G.detach().cpu().numpy()},'
                    txt = txt + f'{lr_D[0]},{loss_D.detach().cpu().numpy()}'
                    txt = txt + f'loss_D_real: {loss_D_real.detach().cpu().numpy()}'
                    txt = txt + f'loss_D_fake: {loss_D_fake.detach().cpu().numpy()}\n'
                    fp.write(txt)
            
            if total_step%print_freq==0 or total_step==1:
                rest_step = epoch*len(train_loader)-total_step
                time_per_step = int(time.time()-start_time) / total_step

                elapsed = datetime.timedelta(seconds=int(time.time()-start_time))
                eta = datetime.timedelta(seconds=int(rest_step*time_per_step))
                lg = f'{total_step}/{len(train_loader)*epoch}, Epoch:{e:03}, elepsed: {elapsed}, '
                lg = lg + f'eta: {eta}, loss_G: {loss_G:f}, loss_D: {loss_D:f}, '
                lg = lg + f'loss_D_real: {loss_D_real:f}, loss_D_fake: {loss_D_fake:f}'
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
                        fake_val = netG(P1,map1,map2)
                    
                    input_val = tensor2ndarray(P1)
                    fake_val = tensor2ndarray(fake_val)
                    real_val = tensor2ndarray(P2)
                    psnr_fake += calculate_psnr(fake_val, real_val, crop_border=4, test_y_channel=False)
                    ssim_fake += calculate_ssim(fake_val, real_val, crop_border=4, test_y_channel=False)
                    
                    if total_step%eval_freq==0:
                        os.makedirs(os.path.join(image_out_dir, f'{total_step:06}'), exist_ok=True)

                        input_val = Image.fromarray(input_val)
                        fake_val = Image.fromarray(fake_val)
                        real_val = Image.fromarray(real_val)
                        img = Image.new('RGB', size=(3*input_val.width, input_val.height), color=0)
                        img.paste(input_val, box=(0, 0))
                        img.paste(fake_val, box=(input_val.width, 0))
                        img.paste(real_val, box=(2*input_val.width, 0))
                        img.save(os.path.join(image_out_dir, f'{total_step:06}', f'{j:03}.jpg'), 'JPEG')
                    
                    val_step += 1
                    if val_step==100: break

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
    PARAMS = {'epoch': 10,
              'batchsize': 8, # each GPU's batch_size is BATCHSIZE/#GPU 
              'model_name': 'simple_02', 
              'print_freq': 100, 
              'eval_freq': 1000}
    torch.backends.cudnn.benchmark = True
    train(epoch=PARAMS['epoch'], 
          batch_size=PARAMS['batchsize'], 
          model_name=PARAMS['model_name'], 
          print_freq=PARAMS['print_freq'], 
          eval_freq=PARAMS['eval_freq'])