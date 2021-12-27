import argparse
import os

from os import listdir
from os.path import isfile, join
import random
import math
import pickle

from tqdm import tqdm

import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shift_right(x) :
  return torch.roll(x,1,2)

def shift_down(x) :
  return torch.roll(x,1,1)

def create_mask(batch_size, W) :
  """
  Causal masking is employed by setting all A_m,n = 0
  where n > m during self-attention
  """
  mask = np.tril(np.ones((batch_size,W,W)),k=0).astype("uint8")
  return torch.Tensor(mask).int()

def positionalencoding2d(d_model, height, width, batch_size):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :param batch_size: size of the batch
    :return: batch_size * height * width * d_model position matrix
    :source: https://github.com/wzlxjtu/PositionalEncoding2D
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe.permute(1,2,0).repeat(batch_size,1,1,1)

def delete_gray_img(root) :
  list_img = [join(root, f) for f in listdir(root) if isfile(join(root, f))]
  nb_image_del = 0

  for p in list_img :
    img = Image.open(p)
    if(len(np.array(img).shape) != 3):
      os.remove(p)
      nb_image_del += 1

  print(nb_image_del,"grayscale images have been removed")

def resize_img_dataset(root_src, root_dst, size) :
  list_img = [f for f in listdir(root_src) if isfile(join(root_src, f))]

  transform_reshape = transforms.Compose([             
      transforms.Resize(size), # interpolaition = BILINEAR
      #transforms.CenterCrop(size)
  ])

  for f in tqdm(list_img) :
    if not os.path.exists(join(root_dst, f)) and f.split(".")[1] == "jpg" :
        img = Image.open(join(root_src, f))
        img_reshape = transform_reshape(img)
        img_reshape.save(join(root_dst, f))
        
      # self.transform_x64_c = transforms.Compose([
    #     transforms.Resize(64), # interpolaition = BILINEAR
    #     transforms.CenterCrop(64),
    #     transforms.Lambda(lambda x : convertTo3bit(x,7)),
    # ])