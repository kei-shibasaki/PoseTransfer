import json
import os
import numpy as np
import torch

def load_option(opt_path):
    with open(opt_path, 'r') as json_file:
        json_obj = json.load(json_file)
        return json_obj

def tensor2ndarray(tensor):
    # Pytorch Tensor (B, C, H, W), [0, 1] -> ndarray (B, H, W, C) [0, 255]
    img = tensor.detach()
    img = img.cpu().permute(0,2,3,1).numpy()
    img = np.clip(img, a_min=0, a_max=1.0)
    img = (img*255).astype(np.uint8)
    return img

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def convert_bits(x, n_bits_out=8, n_bits_in=8):
    """Quantize / dequantize from n_bits_in to n_bits_out."""
    if n_bits_in == n_bits_out:
        return x
    x = x.to(torch.float32)
    x = x / 2**(n_bits_in - n_bits_out)
    x = x.to(torch.int32)
    return x


def nats_to_bits(nats):
    return nats / np.log(2)

def labels_to_bins(labels, num_symbols_per_channel):
    """Maps each (R, G, B) channel triplet to a unique bin.
    Args:
    labels: 4-D Tensor, shape=(batch_size, 3, H, W).
    num_symbols_per_channel: number of symbols per channel.
    Returns:
    labels: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
    """
    b, c, h, w = labels.shape
    labels = labels.to(torch.float32)
    channel_hash = torch.tensor([num_symbols_per_channel**2, num_symbols_per_channel, 1.0]).to(labels.device)
    channel_hash = torch.tile(channel_hash.reshape(1,-1,1,1), dims=(b,1,h,w))
    labels = labels * channel_hash

    labels = torch.sum(labels, dim=1)
    labels = labels.to(torch.int64)
    return labels

def bins_to_labels(bins, num_symbols_per_channel):
    """Maps back from each bin to the (R, G, B) channel triplet.
    Args:
        bins: 3-D Tensor, shape=(batch_size, H, W) with 512 possible symbols.
        num_symbols_per_channel: number of symbols per channel.
    Returns:
        labels: 4-D Tensor, shape=(batch_size, 3, H, W)
    """
    labels = []
    factor = int(num_symbols_per_channel**2)

    for _ in range(3):
        channel = (bins/factor).floor()
        labels.append(channel)

        bins = torch.fmod(bins, factor)
        factor = factor // num_symbols_per_channel
    return torch.stack(labels, dim=1)

