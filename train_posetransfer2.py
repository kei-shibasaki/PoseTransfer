import torch
from torch.nn.modules import loss
import torch.utils
from torch import nn
import os
import time
from PIL import Image
import datetime

from model.posetransfer_model import PoseTransferModel
from dataloader import DeepFashionTrainDataset, DeepFashionValDataset
from utils import tensor2ndarray, load_option
from metrics import calculate_psnr, calculate_ssim

def train(opt_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt = load_option(opt_path)
    
    epoch = opt["PoseTransfer"]['train']['epoch']
    batch_size_train = opt["PoseTransfer"]['train']['batch_size']
    batch_size_val = opt["PoseTransfer"]['val']['batch_size']
    print_freq = opt["PoseTransfer"]['train']['print_freq']
    eval_freq = opt["PoseTransfer"]['train']['eval_freq']
    model_name = opt['name']
    
    model_ckpt_dir = f'./experiments/{model_name}/ckpt'
    image_out_dir = f'./experiments/{model_name}/generated'
    log_dir = f'./experiments/{model_name}/logs'
    os.makedirs(model_ckpt_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    with open(f'{log_dir}/log_{model_name}.log', mode='w', encoding='utf-8') as fp:
        fp.write(f'epoch: {epoch}, batch_size: {batch_size_train}, model_name: {model_name}\n')
    with open(f'{log_dir}/losses_{model_name}.csv', mode='w', encoding='utf-8') as fp:
        fp.write('step,lr_g,loss_g,lr_d,loss_d,loss_d_real,loss_d_fake\n')
    
    netG = PoseTransferModel(opt['PoseTransfer']).to(device)

    optimG = torch.optim.Adam(netG.parameters(), lr=opt["PoseTransfer"]['train']['init_lr'])
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(
        optimG, milestones=opt['PoseTransfer']['train']['milestones'], gamma=opt['PoseTransfer']['train']['gamma'])
    
    train_dataset = DeepFashionTrainDataset(opt['PoseTransfer']['train']['resolution'], opt['dataset_path'])
    val_dataset = DeepFashionValDataset(opt['PoseTransfer']['train']['resolution'], opt['dataset_path'])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_val, shuffle=False, num_workers=2)
    
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
            logits = netG(P1, map1, P2, map2)
            loss_G = netG.loss(logits, P2)
            
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
                        z = netG.encoder(P1, map1, map2)
                        fake_vals, _ = netG.autoregressive_sample(z)
                    
                    for b in range(fake_vals.shape[0]):
                        input_val = tensor2ndarray(P1[b,:,:,:].unsqueeze(0))
                        fake_val = tensor2ndarray(fake_vals[b,:,:,:].unsqueeze(0))
                        real_val = tensor2ndarray(P2[b,:,:,:].unsqueeze(0))
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
                            img.save(os.path.join(image_out_dir, f'{total_step:06}', f'{val_step:04}.jpg'), 'JPEG')
                        
                        val_step += 1
                        if val_step==opt['PoseTransfer']['val']['val_step']: break
                    if val_step==opt['PoseTransfer']['val']['val_step']: break

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
    train('./config/config.json')