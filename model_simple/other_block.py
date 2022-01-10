import torch 
from torch import nn

class Shift(nn.Module):
    def __init__(self, mode):
        super(Shift, self).__init__()
        assert mode in ['down', 'right']
        self.mode = mode
        self.dim = 2 if mode=='down' else 3
    
    def forward(self, x):
        x = torch.roll(x, shifts=1, dims=self.dim)
        if self.mode=='down':
            x[:,:,0,:] = 0.0
        else:
            x[:,:,:,0] = 0.0
        
        return x

