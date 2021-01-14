
"""
paper: Pyramid Channel-based Feature Attention Network for image dehazing 
file: datasets.py
about: build the training dataset and testing dataset
author: Tao Wang
date: 01/13/21
"""

# --- Imports --- #
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import random
import torchvision


class DehazingDataset(Dataset):
    def __init__(self, root_dir, crop=False, crop_size=128, multi_scale=False, rotation=False, color_augment=False, transform=None, train = True):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        if train:
            data_list = root_dir + 'train.txt'
        else:
            data_list = root_dir + 'val.txt'

        with open(data_list) as f:
            contents = f.readlines()
            haze_image_files = [i.strip() for i in contents]
            clear_image_files = [i.split('_')[0] + '.png' for i in haze_image_files]    
        
        self.hazy_image_files = haze_image_files
       
        self.clear_image_files = clear_image_files
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.multi_scale = multi_scale
        self.rotation = rotation
        self.color_augment = color_augment
        self.rotate90 = transforms.RandomRotation(90)  
        self.rotate45 = transforms.RandomRotation(45)    
        self.train = train
    def __len__(self):
        return len(self.hazy_image_files)

    def __getitem__(self, idx):

        haze_name = self.hazy_image_files[idx]
        clear_name = self.clear_image_files[idx]
        hazy_image = Image.open(self.root_dir + 'hazy/' + haze_name).convert('RGB')
        clear_image = Image.open(self.root_dir + 'clear/' + clear_name).convert('RGB')

        if self.rotation:
            degree = random.choice([90, 180, 270])
            hazy_image = transforms.functional.rotate(hazy_image, degree) 
            clear_image = transforms.functional.rotate(clear_image, degree)

        if self.color_augment:
            hazy_image = transforms.functional.adjust_gamma(hazy_image, 1)
            clear_image = transforms.functional.adjust_gamma(clear_image, 1)                           
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            hazy_image = transforms.functional.adjust_saturation(hazy_image, sat_factor)
            clear_image = transforms.functional.adjust_saturation(clear_image, sat_factor)
            
        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)

        if self.crop:
            W = hazy_image.size()[1]
            H = hazy_image.size()[2] 

            Ws = np.random.randint(0, W-self.crop_size-1, 1)[0]
            Hs = np.random.randint(0, H-self.crop_size-1, 1)[0]
            
            hazy_image = hazy_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
            clear_image = clear_image[:,Ws:Ws+self.crop_size,Hs:Hs+self.crop_size]
                       
        if self.multi_scale:
            H = clear_image.size()[1]
            W = clear_image.size()[2]
            hazy_image_s1 = transforms.ToPILImage()(hazy_image)
            clear_image_s1 = transforms.ToPILImage()(clear_image)
            hazy_image_s2 = transforms.ToTensor()(transforms.Resize([H/2, W/2])(hazy_image_s1))
            clear_image_s2 = transforms.ToTensor()(transforms.Resize([H/2, W/2])(clear_image_s1))
            hazy_image_s3 = transforms.ToTensor()(transforms.Resize([H/4, W/4])(hazy_image_s1))
            clear_image_s3 = transforms.ToTensor()(transforms.Resize([H/4, W/4])(clear_image_s1))
            hazy_image_s1 = transforms.ToTensor()(hazy_image_s1)
            clear_image_s1 = transforms.ToTensor()(clear_image_s1)
            return {'hazy_image_s1': hazy_image_s1, 'hazy_image_s2': hazy_image_s2, 'hazy_image_s3': hazy_image_s3, 'clear_image_s1': clear_image_s1, 'clear_image_s2': clear_image_s2, 'clear_image_s3': clear_image_s3}
        else:
            if self.train:
                return {'hazy_image': hazy_image, 'clear_image': clear_image}
            else:
                return {'hazy_image': hazy_image, 'clear_image': clear_image, 'haze_name': haze_name}
