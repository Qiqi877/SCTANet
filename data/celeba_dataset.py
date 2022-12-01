import os
import random
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as tf

from data.base_dataset import BaseDataset


class CelebADataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.shuffle = True if opt.phase=='train' else False 
        self.lr_size = opt.load_size // opt.scale_factor
        self.hr_size = opt.load_size
        self.istrain = opt.phase
        if opt.phase=='train':
            self.img_dir = opt.dataroot_train
        else:
            self.img_dir = opt.dataroot_test
        self.img_names = self.get_img_names()

        self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                Scale((1.0, 1.3), opt.load_size) 
                ])

        self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def get_img_names(self,):
        img_names = [x for x in os.listdir(self.img_dir)] 
        if self.shuffle:
            random.shuffle(img_names)
        return img_names

    def __len__(self,):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])

        hr_img = Image.open(img_path).convert('RGB')
        if self.istrain=='train':
            hr_img = self.aug(hr_img)

        # downsample and upsample to get the LR image
        lr_img = hr_img.resize((self.lr_size, self.lr_size), Image.BICUBIC) 

        hr_tensor = self.to_tensor(hr_img)
        lr_tensor = self.to_tensor(lr_img)

        return {'HR': hr_tensor, 'LR': lr_tensor, 'HR_paths': img_path}


class Scale():
    """
    Random scale the image and pad to the same size if needed.
    ---------------
    # Args:
        factor: tuple input, max and min scale factor.        
    """
    def __init__(self, factor, size):
        self.factor = factor 
        rc_scale = (2 - factor[1], 1)
        self.size   = (size, size)
        self.rc_scale = rc_scale
        self.ratio = (3. / 4., 4. / 3.) 
        self.resize_crop = transforms.RandomResizedCrop(size, rc_scale)

    def __call__(self, img):
        scale_factor = random.random() * (self.factor[1] - self.factor[0]) + self.factor[0]  
        w, h = img.size
        sw, sh = int(w*scale_factor), int(h*scale_factor)
        scaled_img = tf.resize(img, (sh, sw))
        if sw >= w:
            i, j, h, w = self.resize_crop.get_params(img, self.rc_scale, self.ratio)
            scaled_img = tf.resized_crop(img, i, j, h, w, self.size, Image.BICUBIC) 
        elif sw < w:
            lp = (w - sw) // 2
            tp = (h - sh) // 2 
            padding = (lp, tp, w - sw - lp, h - sh - tp) 
            scaled_img = tf.pad(scaled_img, padding)
        return scaled_img 

