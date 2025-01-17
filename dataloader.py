import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import functional as TF
import torchvision
import numpy as np
import json
import glob
import os
import pandas as pd
from PIL import Image
import random

MISSING_VALUE = -1

def load_pose_cords_from_strings(y_str, x_str):
    y_cords = json.loads(y_str)
    x_cords = json.loads(x_str)
    return np.concatenate([np.expand_dims(y_cords, -1), np.expand_dims(x_cords, -1)], axis=1)

def cords_to_map(cords, img_size, old_size=None, affine_matrix=None, sigma=6):
    old_size = img_size if old_size is None else old_size
    cords = cords.astype(float)
    #print(img_size, cords.shape[0:1])
    result = np.zeros(list(img_size) + list(cords.shape[0:1]), dtype='float32')
    for i, point in enumerate(cords):
        if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
            continue
        point[0] = point[0]/old_size[0] * img_size[0]
        point[1] = point[1]/old_size[1] * img_size[1]
        if affine_matrix is not None:
            point_ =np.dot(affine_matrix, np.matrix([point[1], point[0], 1]).reshape(3,1))
            point_0 = int(point_[1])
            point_1 = int(point_[0])
        else:
            point_0 = int(point[0])
            point_1 = int(point[1])
        xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
        result[..., i] = np.exp(-((yy - point_0) ** 2 + (xx - point_1) ** 2) / (2 * sigma ** 2))
    return result

def keypoints2posemap(ann, img_size=(256,256)):
    array = load_pose_cords_from_strings(ann['keypoints_y'], ann['keypoints_x'])
    posemap = cords_to_map(array, img_size)
    posemap = np.transpose(posemap, (2, 0, 1))
    posemap = torch.Tensor(posemap)
    return posemap

class DeepFashionTrainDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        '''
        directories or files using for creating dataset
        train_256
        fasion-annotation-train.csv
        fasion-pairs-train.csv
        '''
        super(DeepFashionTrainDataset, self).__init__()
        self.opt = opt
        self.res = opt.input_image_resolution
        self.pose_res = opt.input_pose_resolution
        
        self.dataset_path = opt.dataset_path
        self.annfile = pd.read_csv(os.path.join(self.dataset_path, 'fasion-annotation-train.csv'), sep=':')
        self.annfile = self.annfile.set_index('name')
        self.pairlist = pd.read_csv(os.path.join(self.dataset_path, 'fasion-pairs-train.csv'), sep=',', index_col=0)
        
        self.interp_mode_img = torchvision.transforms.InterpolationMode.BICUBIC
        # ValueError: Interpolation mode 'bicubic' is unsupported with Tensor input
        self.interp_mode_pose = torchvision.transforms.InterpolationMode.NEAREST
    
    def trasnform_images(self, source, target, input_pose, target_pose):
        a_min, a_max = self.opt.random_rotate_range
        angle = random.random() * (a_max-a_min) + a_min
        use_flip = random.random()>0.5 if self.opt.use_random_horizontal_flip else False
        source = TF.resize(source, self.res, self.interp_mode_img)
        target = TF.resize(target, self.res, self.interp_mode_img)
        input_pose = TF.resize(input_pose, self.pose_res, self.interp_mode_pose)
        target_pose = TF.resize(target_pose, self.pose_res, self.interp_mode_pose)
        if use_flip:
            source = TF.hflip(source)
            target = TF.hflip(target)
            input_pose = TF.hflip(input_pose)
            target_pose = TF.hflip(target_pose)
        source = TF.rotate(source, angle, self.interp_mode_img)
        target = TF.rotate(target, angle, self.interp_mode_img)
        input_pose = TF.rotate(input_pose, angle, self.interp_mode_pose)
        target_pose = TF.rotate(target_pose, angle, self.interp_mode_pose)
        
        source = TF.to_tensor(source)
        target = TF.to_tensor(target)
        
        return source, target, input_pose, target_pose
    
    def __len__(self):
        return len(self.pairlist)
    
    def __getitem__(self, idx):
        row = self.pairlist.iloc[idx]
        P1_name, P2_name = row['from'], row['to']
        P1_path = os.path.join(self.dataset_path, 'train_256', P1_name)
        P2_path = os.path.join(self.dataset_path, 'train_256', P2_name)
        
        P1, P2 = Image.open(P1_path), Image.open(P2_path)
        ann_P1, ann_P2 = self.annfile.loc[P1_name], self.annfile.loc[P2_name]
        ann_P1 = keypoints2posemap(ann_P1, img_size=self.pose_res)
        ann_P2 = keypoints2posemap(ann_P2, img_size=self.pose_res)
        
        P1, P2, ann_P1, ann_P2 = self.trasnform_images(P1, P2, ann_P1, ann_P2)
        
        return {'P1': P1, 'P2': P2, 'map1': ann_P1, 'map2': ann_P2, 
                'P1_path': P1_path, 'P2_path': P2_path}

class DeepFashionValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        '''
        directories or files using for creating dataset
        test_256
        fasion-annotation-test.csv
        fasion-pairs-test.csv
        '''
        super(DeepFashionValDataset, self).__init__()
        self.opt = opt
        self.res = opt.input_image_resolution
        self.pose_res = opt.input_pose_resolution
        
        self.dataset_path = opt.dataset_path
        self.annfile = pd.read_csv(os.path.join(self.dataset_path, 'fasion-annotation-test.csv'), sep=':')
        self.annfile = self.annfile.set_index('name')
        self.pairlist = pd.read_csv(os.path.join(self.dataset_path, 'fasion-pairs-test.csv'), sep=',', index_col=0)
        
        self.interp_mode_img = torchvision.transforms.InterpolationMode.BICUBIC
        # ValueError: Interpolation mode 'bicubic' is unsupported with Tensor input
        self.interp_mode_pose = torchvision.transforms.InterpolationMode.NEAREST
        
    def trasnform_images(self, source, target, input_pose, target_pose):
        source = TF.resize(source, self.res, self.interp_mode_img)
        target = TF.resize(target, self.res, self.interp_mode_img)
        input_pose = TF.resize(input_pose, self.pose_res, self.interp_mode_pose)
        target_pose = TF.resize(target_pose, self.pose_res, self.interp_mode_pose)
        source = TF.to_tensor(source)
        target = TF.to_tensor(target)
        
        return source, target, input_pose, target_pose
    
    def __len__(self):
        return len(self.pairlist)
    
    def __getitem__(self, idx):
        row = self.pairlist.iloc[idx]
        P1_name, P2_name = row['from'], row['to']
        P1_path = os.path.join(self.dataset_path, 'test_256', P1_name)
        P2_path = os.path.join(self.dataset_path, 'test_256', P2_name)
        
        P1, P2 = Image.open(P1_path), Image.open(P2_path)
        ann_P1, ann_P2 = self.annfile.loc[P1_name], self.annfile.loc[P2_name]
        ann_P1 = keypoints2posemap(ann_P1, img_size=self.pose_res)
        ann_P2 = keypoints2posemap(ann_P2, img_size=self.pose_res)
        
        P1, P2, ann_P1, ann_P2 = self.trasnform_images(P1, P2, ann_P1, ann_P2)
        
        return {'P1': P1, 'P2': P2, 'map1': ann_P1, 'map2': ann_P2, 
                'P1_path': P1_path, 'P2_path': P2_path}


class Market1501TrainDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        '''
        directories or files using for creating dataset
        train_12864
        fasion-annotation-train.csv
        fasion-pairs-train.csv
        '''
        super(Market1501TrainDataset, self).__init__()
        self.opt = opt
        self.res = opt.input_image_resolution
        self.pose_res = opt.input_pose_resolution
        
        self.dataset_path = opt.dataset_path
        self.annfile = pd.read_csv(os.path.join(self.dataset_path, 'market-annotation-train.csv'), sep=':')
        self.annfile = self.annfile.set_index('name')
        self.pairlist = pd.read_csv(os.path.join(self.dataset_path, 'market-pairs-train.csv'), sep=',')
        
        self.interp_mode_img = torchvision.transforms.InterpolationMode.BICUBIC
        # ValueError: Interpolation mode 'bicubic' is unsupported with Tensor input
        self.interp_mode_pose = torchvision.transforms.InterpolationMode.NEAREST
        
    def trasnform_images(self, source, target, input_pose, target_pose):
        a_min, a_max = self.opt.random_rotate_range
        angle = random.random() * (a_max-a_min) + a_min
        use_flip = random.random()>0.5 if self.opt.use_random_horizontal_flip else False
        source = TF.resize(source, self.res, self.interp_mode_img)
        target = TF.resize(target, self.res, self.interp_mode_img)
        input_pose = TF.resize(input_pose, self.pose_res, self.interp_mode_pose)
        target_pose = TF.resize(target_pose, self.pose_res, self.interp_mode_pose)
        if use_flip:
            source = TF.hflip(source)
            target = TF.hflip(target)
            input_pose = TF.hflip(input_pose)
            target_pose = TF.hflip(target_pose)
        source = TF.rotate(source, angle, self.interp_mode_img)
        target = TF.rotate(target, angle, self.interp_mode_img)
        input_pose = TF.rotate(input_pose, angle, self.interp_mode_pose)
        target_pose = TF.rotate(target_pose, angle, self.interp_mode_pose)
        
        source = TF.to_tensor(source)
        target = TF.to_tensor(target)
        
        return source, target, input_pose, target_pose
    
    def __len__(self):
        return len(self.pairlist)
    
    def __getitem__(self, idx):
        row = self.pairlist.iloc[idx]
        P1_name, P2_name = row['from'], row['to']
        P1_path = os.path.join(self.dataset_path, 'train_12864', P1_name)
        P2_path = os.path.join(self.dataset_path, 'train_12864', P2_name)
        
        P1, P2 = Image.open(P1_path), Image.open(P2_path)
        ann_P1, ann_P2 = self.annfile.loc[P1_name], self.annfile.loc[P2_name]
        ann_P1 = keypoints2posemap(ann_P1, img_size=self.pose_res)
        ann_P2 = keypoints2posemap(ann_P2, img_size=self.pose_res)
        
        P1, P2, ann_P1, ann_P2 = self.trasnform_images(P1, P2, ann_P1, ann_P2)
        
        return {'P1': P1, 'P2': P2, 'map1': ann_P1, 'map2': ann_P2, 
                'P1_path': P1_path, 'P2_path': P2_path}

class Market1501ValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        '''
        directories or files using for creating dataset
        test_12864
        fasion-annotation-test.csv
        fasion-pairs-test.csv
        '''
        super(Market1501ValDataset, self).__init__()
        self.opt = opt
        self.res = opt.input_image_resolution
        self.pose_res = opt.input_pose_resolution
        
        self.dataset_path = opt.dataset_path
        self.annfile = pd.read_csv(os.path.join(self.dataset_path, 'market-annotation-test.csv'), sep=':')
        self.annfile = self.annfile.set_index('name')
        self.pairlist = pd.read_csv(os.path.join(self.dataset_path, 'market-pairs-test.csv'), sep=',')
        
        self.interp_mode_img = torchvision.transforms.InterpolationMode.BICUBIC
        # ValueError: Interpolation mode 'bicubic' is unsupported with Tensor input
        self.interp_mode_pose = torchvision.transforms.InterpolationMode.NEAREST
        
    def trasnform_images(self, source, target, input_pose, target_pose):
        source = TF.resize(source, self.res, self.interp_mode_img)
        target = TF.resize(target, self.res, self.interp_mode_img)
        input_pose = TF.resize(input_pose, self.pose_res, self.interp_mode_pose)
        target_pose = TF.resize(target_pose, self.pose_res, self.interp_mode_pose)
        source = TF.to_tensor(source)
        target = TF.to_tensor(target)
        
        return source, target, input_pose, target_pose
    
    def __len__(self):
        return len(self.pairlist)
    
    def __getitem__(self, idx):
        row = self.pairlist.iloc[idx]
        P1_name, P2_name = row['from'], row['to']
        P1_path = os.path.join(self.dataset_path, 'test_12864', P1_name)
        P2_path = os.path.join(self.dataset_path, 'test_12864', P2_name)
        
        P1, P2 = Image.open(P1_path), Image.open(P2_path)
        ann_P1, ann_P2 = self.annfile.loc[P1_name], self.annfile.loc[P2_name]
        ann_P1 = keypoints2posemap(ann_P1, img_size=self.pose_res)
        ann_P2 = keypoints2posemap(ann_P2, img_size=self.pose_res)
        
        P1, P2, ann_P1, ann_P2 = self.trasnform_images(P1, P2, ann_P1, ann_P2)
        
        return {'P1': P1, 'P2': P2, 'map1': ann_P1, 'map2': ann_P2, 
                'P1_path': P1_path, 'P2_path': P2_path}

# This Dataset is used for evaluating inception score
class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, resolution=(256, 256), normalize=False):
        super(SimpleImageDataset, self).__init__()
        self.images = sorted(glob.glob(os.path.join(image_path, '*.jpg')))
        self.normalize = normalize
        self.transform = transforms.Compose([
            transforms.Resize(resolution), 
            transforms.ToTensor()
        ])
        self.mean = torch.Tensor([0.485, 0.456, 0.406])
        self.std = torch.Tensor([0.229, 0.224, 0.225])
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        img = self.transform(img)
        if self.normalize:
            img = torchvision.transforms.functional(img, self.mean, self.std)

        return img
    
    def __len__(self):
        return len(self.images)