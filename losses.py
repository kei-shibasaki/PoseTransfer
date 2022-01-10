import re
import torch.nn.functional as F
import torch
import torchvision

class GANLoss:
    def __init__(self, r=100.0, eps=1e-12, method='bce'):
        assert method in ['bce', 'hinge', 'lsgan']
        self.r = r
        self.eps = eps
        self.method = method
        
        self.loss_fn = {}
        self.loss_fn['bce'] = self.bce_loss
        self.loss_fn['hinge'] = self.hinge_loss
        self.loss_fn['lsgan'] = self.lsgan_loss
    
    def bce_loss(self, logits, mode):
        assert mode in ['gen', 'disc_r', 'disc_f']
        p = F.sigmoid(logits)
        if mode=='gen':
            return -torch.log(p+self.eps).mean()
        elif mode=='disc_r':
            return -torch.log(p+self.eps).mean()
        elif mode=='disc_f':
            return -torch.log(1-p+self.eps).mean()
        else:
            raise NotImplementedError
    
    def hinge_loss(self, logits, mode):
        assert mode in ['gen', 'disc_r', 'disc_f']
        zeros = torch.zeros_like(logits, device=logits.device)
        if mode=='gen':
            return -logits.mean()
        elif mode=='disc_r':
            return -torch.minimum(logits-1, zeros).mean()
        elif mode=='disc_f':
            return -torch.minimum(-logits-1, zeros).mean()
        else:
            raise NotImplementedError
    
    def lsgan_loss(self, logits, mode):
        assert mode in ['gen', 'disc_r', 'disc_f']
        if mode=='gen':
            return 0.5*logits.pow(2).mean()
        elif mode=='disc_r':
            return 0.5*(logits-1).pow(2).mean()
        elif mode=='disc_f':
            return 0.5*(logits+1).pow(2).mean()
        else:
            raise NotImplementedError
    
    def mae_loss(self, image, target):
        return (target-image).abs().mean()
    
    def generator_loss(self, image, target, logits):
        return (self.loss_fn[self.method](logits, 'gen') + self.r * self.mae_loss(image, target)) / self.r
    
    def discriminator_loss_real(self, logits):
        return self.loss_fn[self.method](logits, 'disc_r')
    
    def discriminator_loss_fake(self, logits):
        return self.loss_fn[self.method](logits, 'disc_f')

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

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