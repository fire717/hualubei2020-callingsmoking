from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
import random
from torch.utils.data.dataset import Dataset
import cv2
import torchvision.transforms as transforms

from PIL import Image
from torchsummary import summary
import torchvision.models as models
import pretrainedmodels
from PIL import Image

import albumentations as A              
# from data import TrainValDataAug


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


my_seed = 42
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.cuda.manual_seed(my_seed)

def getAllName(file_dir, tail_list = ['.png','.jpg']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L



class MyDataAug:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        raw_h, raw_w = img.shape[:2]
        min_size = max(img.shape[:2])

        #大图随机旋转
        #if min_size>2000 and img.shape[0]>img.shape[1]:
        # if img.shape[0]>img.shape[1]:
        #     A.ShiftScaleRotate(
        #         shift_limit=0.1,
        #         scale_limit=0.1,
        #         rotate_limit=20,
        #         interpolation=cv2.INTER_LINEAR,
        #         border_mode=cv2.BORDER_CONSTANT,
        #          value=0, mask_value=0,
        #         p=0.6),


        input_h, input_w = img.shape[:2]
        h_ratio = input_h/self.h
        w_ratio = input_w/self.w
        if (h_ratio>1 and w_ratio>1):
            if h_ratio>w_ratio:
                resize_h = self.h
                resize_w = int(input_w/h_ratio)
            else:
                resize_w = self.w
                resize_h = int(input_h/w_ratio)
            img = A.Resize(resize_h,resize_w,cv2.INTER_AREA)(image=img)['image']
            min_size = max(img.shape[:2])

        # img = A.PadIfNeeded(min_height=min_size, min_width=min_size, 
        #             border_mode=4, value=0, mask_value=0, 
        #             always_apply=True, p=1.0)(image=img)['image']
        img = A.OneOf([A.PadIfNeeded(min_height=min_size, min_width=min_size, 
                        border_mode=3, value=0, mask_value=0, 
                        always_apply=False, p=0.7),
                    A.PadIfNeeded(min_height=min_size, min_width=min_size, 
                        border_mode=0, value=0, mask_value=0, 
                        always_apply=False, p=0.3)],
                        p=1.0)(image=img)['image']
        

        if raw_h>raw_w and ((raw_w<self.w-2  and random.random()<0.05) or (raw_w>self.w-2 and random.random()<0.15)):
            img = A.OneOf([A.Rotate(limit=[80,90], interpolation=cv2.INTER_LINEAR, 
                        border_mode=3, p=0.6),
                        A.Rotate(limit=[-80,-90], interpolation=cv2.INTER_LINEAR, 
                        border_mode=3, p=0.4)],
                            p=1)(image=img)['image']

        else:
            img = A.ShiftScaleRotate(
                                    shift_limit=0.1,
                                    scale_limit=0.1,
                                    rotate_limit=20,
                                    interpolation=cv2.INTER_LINEAR,
                                    border_mode=cv2.BORDER_CONSTANT,
                                     value=0, mask_value=0,
                                    p=0.6)(image=img)['image']

        img = A.HorizontalFlip(p=0.5)(image=img)['image'] 
        
        img = A.OneOf([A.RandomBrightness(limit=0.1, p=1), 
                    A.RandomContrast(limit=0.1, p=1),
                    A.RandomGamma(gamma_limit=(50, 150),p=1),
                    A.HueSaturationValue(hue_shift_limit=10, 
                        sat_shift_limit=10, val_shift_limit=10,  p=1)], 
                    p=0.8)(image=img)['image']

        
        img = A.Resize(self.h,self.w,p=1)(image=img)['image']
        img = A.OneOf([A.MotionBlur(blur_limit=3, p=0.2), 
                        A.MedianBlur(blur_limit=3, p=0.2), 
                        A.GaussianBlur(blur_limit=3, p=0.1),
                        A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.5)], 
                        p=0.8)(image=img)['image']


        
        img = Image.fromarray(img)
        return img

img_size = 600
img_path_list = getAllName("../../data/train/t")[:200]
print(len(img_path_list))
transform = transforms.Compose([

                            MyDataAug(img_size,img_size),

                             
                            #transforms.ToTensor(),
                             #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                             ])


for i,img_path in enumerate(img_path_list):
    #img = Image.open(img_path).convert('RGB')
    img = cv2.imread(img_path)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = transform(img)
    img.save("tmp/"+os.path.basename(img_path), quality=100)

    # if i>100:
    #     break

