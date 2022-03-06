import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_losses():
    os.makedirs('loss_plots', exist_ok=True)
    log_dir_dict = {
        'market_small_pre': 'experiments/market_small_pre/logs'
    }
    
    fig = plt.figure()
    for model_name, log_dir in log_dir_dict.items():
        train_log = pd.read_csv(os.path.join(log_dir, f'train_losses_{model_name}.csv'))
        test_log = pd.read_csv(os.path.join(log_dir, f'test_losses_{model_name}.csv'))
        plt.plot(train_log['step'], train_log['loss_G'], label=f'train_loss_G: {model_name}', alpha=0.6)
        plt.plot(test_log['step'], test_log['loss_G'], label=f'test_loss_G: {model_name}', alpha=0.6)
    
    plt.savefig('loss_plots/losses.png')

if __name__=='__main__':
    pass