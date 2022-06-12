import torch.nn.functional as F
import torch
from torch import nn
import torchvision
from model.misc_blocks import VGG19

class VGGLoss(nn.Module):
    def __init__(self, resize=False, normalize=False, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(VGGLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.resize = resize
        self.normalize = normalize
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G
        
    def __call__(self, x, y):
        # Compute features
        if self.resize:
            input = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(y, mode='bilinear', size=(224, 224), align_corners=False)
        if self.normalize:
            x = (x-self.mean) / self.std
            y = (y-self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))


        return content_loss, style_loss

def gradient_penalty(netD, real, fake):
    b_size = real.size(0)
    alpha = torch.rand(b_size,1,1,1).to(real.device)
    alpha = alpha.expand_as(real)
    
    interpolation = alpha*real + (1-alpha)*fake
    interpolation = torch.autograd.Variable(interpolation, requires_grad=True)
    logits = netD(interpolation)
    
    grad_outputs = torch.ones_like(logits).to(real.device)
    grads = torch.autograd.grad(
        outputs=logits, inputs=interpolation, 
        grad_outputs=grad_outputs, 
        create_graph=True, retain_graph=True, 
        only_inputs=True)[0]
    grads = grads.view(b_size, -1)
    grad_norm = grads.norm(2,1)
    return torch.mean((grad_norm-1)**2)