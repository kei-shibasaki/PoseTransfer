from email.policy import default
import pandas as pd
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
from easydict import EasyDict
from utils.utils import load_option

def smoothing(array, a):
    new_array = np.zeros_like(array)
    new_array[0] = array[0]
    for i in range(1, len(new_array)):
        new_array[i] = (1-a)*array[i] + a*new_array[i-1]
    return new_array

def plot_losses(opt_paths, a, plot_detail_loss=False):
    for opt_path in opt_paths:
        opt = EasyDict(load_option(opt_path))
        
        model_name = opt.name
        log_dir = os.path.join('experiments', model_name, 'logs')
        train_log = pd.read_csv(os.path.join(log_dir, f'train_losses_{model_name}.csv'))
        test_log = pd.read_csv(os.path.join(log_dir, f'test_losses_{model_name}.csv'))
        
        plt.figure()
        plt.plot(train_log['step'], smoothing(train_log['loss_G'], a), label=f'train_loss_G', alpha=0.75)
        plt.plot(train_log['step'], smoothing(train_log['loss_D'], a), label=f'train_loss_D', alpha=0.75)
        if plot_detail_loss:
            plt.plot(train_log['step'], smoothing(train_log['advloss'], a), label=f'train_advloss', alpha=0.25)
            plt.plot(train_log['step'], smoothing(train_log['l1loss'], a), label=f'train_l1loss', alpha=0.25)
            plt.plot(train_log['step'], smoothing(train_log['ploss'], a), label=f'train_ploss', alpha=0.25)
            plt.plot(train_log['step'], smoothing(train_log['sloss'], a), label=f'train_sloss', alpha=0.25)
            plt.plot(train_log['step'], smoothing(train_log['loss_D_real'], a), label=f'train_loss_D_real', alpha=0.25)
            plt.plot(train_log['step'], smoothing(train_log['loss_D_fake'], a), label=f'train_loss_D_fake', alpha=0.25)
            plt.plot(train_log['step'], smoothing(train_log['gp'], a), label=f'train_gp', alpha=0.25)
        # plt.ylim(top=1.0)
        plt.xlabel('Steps')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.title(f'Loss {model_name}')
        plt.grid(axis='y')
        plt.savefig(os.path.join(log_dir, f'losses_{model_name}.png'))
        
        fig = plt.figure(figsize=(19.2, 4.8))
        
        ax1 = fig.add_subplot(131)
        ax1.plot(test_log['step'], test_log['psnr'], label=f'PSNR', alpha=1.0)
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('PSNR')
        ax1.set_title(f'PSNR {model_name}')
        ax1.grid(axis='y')
        
        ax2 = fig.add_subplot(132)
        ax2.plot(test_log['step'], test_log['ssim'], label=f'SSIM', alpha=1.0)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('SSIM')
        ax2.set_title(f'SSIM {model_name}')
        ax2.grid(axis='y')
        
        ax3 = fig.add_subplot(133)
        ax3.plot(test_log['step'], test_log['lpips'], label=f'LPIPS', alpha=1.0)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('LPIPS')
        ax3.set_title(f'LPIPS {model_name}')
        ax3.grid(axis='y')
        
        plt.savefig(os.path.join(log_dir, f'metrics_{model_name}.png'))
        

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A script of plot losses.')
    parser.add_argument('-c', '--config_files', nargs='+', required=True)
    parser.add_argument('-a', '--smoothing_ratio', default=0.9, type=float)
    parser.add_argument('-p', '--plot_detail_loss', action='store_true')
    
    args = parser.parse_args()
    
    plot_losses(args.config_files, args.smoothing_ratio, args.plot_detail_loss)