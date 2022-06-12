import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from PIL import Image
from tqdm import tqdm
import argparse
from collections import OrderedDict

from utils.utils import tensor2ndarray, load_option
from utils.pose_utils import draw_pose_from_map
from dataloader import DeepFashionValDataset, Market1501ValDataset
from model.pose_transformer import PoseTransformer
from easydict import EasyDict

def generate_images(opt, batch_size, checkpoint_path, out_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    os.makedirs(os.path.join(out_path, 'generated'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'GT'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'comparison'), exist_ok=True)
    
    print('Preparing Data...')
    if opt.dataset_type=='fashion':
        val_dataset = DeepFashionValDataset(opt)
    elif opt.dataset_type=='market':
        val_dataset = Market1501ValDataset(opt)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print('Creating Net...')
    net = PoseTransformer(opt).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    #new_dict = OrderedDict()
    #for key, value in checkpoint['netG_state_dict'].items():
    #    key = key[7:]
    #    new_dict[key] = value
    #net.load_state_dict(new_dict)
    net.load_state_dict(checkpoint['netG_state_dict'])
    
    print('Generating Images...')
    net.eval()
    for i, data_dict in enumerate(tqdm(val_loader)):
        P1, P2, map1, map2, P1_path, P2_path = data_dict.values()
        P1, P2, map1, map2 = P1.to(device), P2.to(device), map1.to(device), map2.to(device)
        b_size, _, _, _ = P1.shape
        
        with torch.no_grad():
            out = net(P1,map1,map2)
            out = out.sigmoid()
        
        input_vals = tensor2ndarray(P1)
        fake_vals = tensor2ndarray(out)
        real_vals = tensor2ndarray(P2)
        
        for b in range(b_size):
            img = out[b,:,:,:]
            mp1 = map1[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
            mp1, _ = draw_pose_from_map(mp1)
            mp2 = map2[b,:,:,:].detach().cpu().permute(1,2,0).numpy()
            mp2, _ = draw_pose_from_map(mp2)

            input_val = Image.fromarray(input_vals[b,:,:,:])
            mp1 = Image.fromarray(mp1)
            fake_val = Image.fromarray(fake_vals[b,:,:,:])
            mp2 = Image.fromarray(mp2)
            real_val = Image.fromarray(real_vals[b,:,:,:])
            
            P1_name = os.path.splitext(os.path.basename(P1_path[b]))[0]
            P2_name = os.path.splitext(os.path.basename(P2_path[b]))[0]
            fname = f'{P1_name}_TO_{P2_name}.jpg'
            
            real_val.save(os.path.join(out_path, 'GT', fname), 'JPEG')
            fake_val.save(os.path.join(out_path, 'generated', fname), 'JPEG')
            
            img = Image.new('RGB', size=(5*input_val.width, input_val.height), color=0)
            img.paste(input_val, box=(0, 0))
            img.paste(mp1, box=(input_val.width, 0))
            img.paste(fake_val, box=(2*input_val.width, 0))
            img.paste(mp2, box=(3*input_val.width, 0))
            img.paste(real_val, box=(4*input_val.width, 0))
            img.save(os.path.join(out_path, 'comparison', fname), 'JPEG')
            

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of generate images.')
    parser.add_argument('-c', '--config', required=True, help='Path of config file')
    parser.add_argument('-b', '--batch_size', default=32, help='Batch size')
    parser.add_argument('-ckpt', '--checkpoint_path', default=None, help='Path to the chenckpoint')
    args = parser.parse_args()
    opt = EasyDict(load_option(args.config))
    
    batch_size = int(args.batch_size)
    
    model_name = opt.name
    if args.checkpoint_path==None:
        checkpoint_path = os.path.join('experiments', model_name, 'ckpt', f'{model_name}_{opt.steps}.ckpt')
    else:
        checkpoint_path = args.checkpoint_path
    
    out_path = f'results/{model_name}'
    
    generate_images(opt, batch_size, checkpoint_path, out_path)