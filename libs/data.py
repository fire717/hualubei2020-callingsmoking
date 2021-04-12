
from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random
import cv2
import albumentations as A
import json
import platform

###### 1.Data aug
class TrainValDataAug:
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
        

        # if raw_h>raw_w and ((raw_w<self.w-2  and random.random()<0.05) or (raw_w>self.w-2 and random.random()<0.15)):
        #     img = A.OneOf([A.Rotate(limit=[80,90], interpolation=cv2.INTER_LINEAR, 
        #                 border_mode=3, p=0.6),
        #                 A.Rotate(limit=[-80,-90], interpolation=cv2.INTER_LINEAR, 
        #                 border_mode=3, p=0.4)],
        #                     p=1)(image=img)['image']

        # else:
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

class TestDataAug:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img):
        # opencv img, BGR
        # new_width, new_height = self.size[0], self.size[1]
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        min_size = max(img.shape[:2])

        # input_h, input_w = img.shape[:2]
        # h_ratio = input_h/self.h
        # w_ratio = input_w/self.w
        # if (h_ratio>1 and w_ratio>1):
        #     if h_ratio>w_ratio:
        #         resize_h = self.h
        #         resize_w = int(input_w/h_ratio)
        #     else:
        #         resize_w = self.w
        #         resize_h = int(input_h/w_ratio)
        #     img = A.Resize(resize_h,resize_w,cv2.INTER_AREA)(image=img)['image']
        #     min_size = max(img.shape[:2])

        

        img = A.PadIfNeeded(min_height=min_size, min_width=min_size, 
                        border_mode=3, value=0, mask_value=0, 
                        always_apply=False, p=1)(image=img)['image']

        # if raw_h>raw_w and ((raw_w<2000 and random.random()<0.05) or (raw_w>2000 and random.random()<0.15)):
        #     img = A.Rotate(limit=[80,90], interpolation=cv2.INTER_LINEAR, 
        #                 border_mode=3, 
        #                  p=1)(image=img)['image']

        img = A.Resize(self.h,self.w,p=1)(image=img)['image']
        img = Image.fromarray(img)
        return img



######## 2.dataloader
class TensorDatasetTestMutil(Dataset):

    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # img = cv2.imread(self.train_jpg[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,(180, 180))

        img = Image.open(self.train_jpg[index]).convert('RGB')
        #img = cv2.imread(self.train_jpg[index])
        #img = imgPaddingWrap(img)
        #b
        if self.transform is not None:
            img = self.transform(img)

        y = 0#np.array([0,0,0])
        if 'call' in self.train_jpg[index]:
            y = 0
        elif  'normal' in self.train_jpg[index]:
            y = 1
        else:
            y = 2
        return img, y, self.train_jpg[index]

    def __len__(self):
        return len(self.train_jpg)

class TensorDatasetTestClassify(Dataset):

    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # img = cv2.imread(self.train_jpg[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,(180, 180))

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])
        #img = imgPaddingWrap(img)
        #b
        if self.transform is not None:
            img = self.transform(img)

        y = 0
        if  'smok' in self.train_jpg[index] and 'call' in self.train_jpg[index]:
            y = 3
        elif  'normal' in self.train_jpg[index]:
            y = 1
        elif  'smok' in self.train_jpg[index]:
            y = 2
        return img, y, self.train_jpg[index]

    def __len__(self):
        return len(self.train_jpg)

class TensorDatasetTestClassifyTTA(Dataset):

    def __init__(self, train_jpg, h,w,transform=None, tta=3):
        self.h = h
        self.w = w
        self.train_jpg = train_jpg
        self.tta = tta
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # img = cv2.imread(self.train_jpg[index])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img,(180, 180))

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])
        #img = imgPaddingWrap(img)
        #b

        img_list = []

        for i in range(self.tta):
            img_new = self.transform(img)
            #img_new.save("a%d.jpg" % i)
            img_list.append(img_new)

        y = 0#np.array([0,0,0])
        if 'call' in self.train_jpg[index] or 'phone' in self.train_jpg[index]:
            y = 0
        elif  'normal' in self.train_jpg[index]:
            y = 1
        else:
            y = 2
        return img_list, y, self.train_jpg[index]

    def __len__(self):
        return len(self.train_jpg)


class TensorDatasetTrainValMutil(Dataset):

    def __init__(self, train_jpg, transform=None):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])

        if self.transform is not None:
            img = self.transform(img)

        y = np.array([0,0], dtype=np.float32)
        if  'smoking_calling' in self.train_jpg[index]:
            #y = 1
            y[0] = 1.0
            y[1] = 1.0
        elif 'calling' in self.train_jpg[index]:
            y[0] = 1.0
        elif  'normal' in self.train_jpg[index]:
            pass
        else:
            y[1] = 1.0
        return img, y
        
    def __len__(self):
        return len(self.train_jpg)

class TensorDatasetTrainValClassify(Dataset):

    def __init__(self, train_jpg, transform=None, distill=False):
        self.train_jpg = train_jpg
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
        self.distill = distill
        if distill:
            distill_path = r"save/good/result_distill.json"
            print("distill path: ", distill_path)
            with open(distill_path,'r') as f:
                self.distill_dict = json.loads(f.readlines()[0])  

    def __getitem__(self, index):

        #img = Image.open(self.train_jpg[index]).convert('RGB')
        img = cv2.imread(self.train_jpg[index])

        if self.transform is not None:
            img = self.transform(img)

        # y = np.array([0,0,0], dtype=np.float32)
        #print(self.train_jpg[index])
        # if 'calling_images' in self.train_jpg[index]:
        y = 0
        if  'smok' in self.train_jpg[index] and 'call' in self.train_jpg[index]:
            y = 3
        elif  'normal' in self.train_jpg[index]:
            y = 1
        elif  'smok' in self.train_jpg[index]:
            y = 2
        # print(y)
        # b
        if self.distill:
            y_onehot = [0,0,0,0]
            y_onehot[y] = 1
            y_onehot = np.array(y_onehot)
            if os.path.basename(self.train_jpg[index]) in self.distill_dict:
                y = y_onehot*0.6+np.array(self.distill_dict[os.path.basename(self.train_jpg[index])])*0.4
            else:
                y = y_onehot*0.9 + (1-0.9)/4
        return img, y
        
    def __len__(self):
        return len(self.train_jpg)


###### 3. get data loader 


def getDataLoader(mode, input_data,model_name, img_size, batch_size, kwargs):
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = 4


    if model_name == 'xception':
        my_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif "adv" in model_name:
        my_normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    elif "resnex" in model_name or 'eff' in model_name or 'RegNet' in model_name:
        my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #my_normalize = transforms.Normalize([0.4783, 0.4559, 0.4570], [0.2566, 0.2544, 0.2522])
    elif "EN-B" in model_name:
        my_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        raise Exception("Not found normalize type!")

    if mode=="trainMutil":
        my_dataloader = TensorDatasetTrainValMutil
        
        train_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[0],transforms.Compose([
                                TrainValDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize,
                                ])),
                        batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[1],transforms.Compose([
                                TestDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize
                                ])),
                        batch_size=batch_size, shuffle=True, **kwargs)
        return train_loader, val_loader

    if mode=="trainClassify":
        my_dataloader = TensorDatasetTrainValClassify
        
        train_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[0],transforms.Compose([
                                TrainValDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize,
                                ])),
                        batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[1],transforms.Compose([
                                TrainValDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize
                                ])),
                        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        return train_loader, val_loader

    if mode=="trainClassifyOnehot":
        my_dataloader = TensorDatasetTrainValClassify
        
        train_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[0],transforms.Compose([
                                TrainValDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize,
                                ]), distill=True),
                        batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
                    my_dataloader(input_data[1],transforms.Compose([
                                TrainValDataAug(img_size, img_size),
                                transforms.ToTensor(),
                                my_normalize
                                ]), distill=True),
                        batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
        return train_loader, val_loader



    elif mode=="test":
        my_dataloader = TensorDatasetTestClassify

        test_loader = torch.utils.data.DataLoader(
                my_dataloader(input_data[0],
                        transforms.Compose([
                                    TestDataAug(img_size, img_size),
                                    transforms.ToTensor(),
                                    my_normalize
                                ])
                ), batch_size=batch_size, shuffle=False, 
                num_workers=kwargs['num_workers'], pin_memory=kwargs['pin_memory']
            )

        return test_loader

    elif mode=="testTTA":
        my_dataloader = TensorDatasetTestClassifyTTA

        test_loader = torch.utils.data.DataLoader(
                my_dataloader(input_data[0],img_size, img_size,
                        transforms.Compose([
                                    TrainValDataAug(img_size, img_size),
                                    transforms.ToTensor(),
                                    my_normalize
                                ]),
                ), batch_size=1, shuffle=False, num_workers=0, pin_memory=False
            )

        return test_loader

    # elif mode=="eval":
    #     my_dataloader = TensorDatasetTestClassify

    #     test_loader = torch.utils.data.DataLoader(
    #             my_dataloader(input_data[0],img_size, img_size,
    #                     transforms.Compose([
    #                                 TrainValDataAug(img_size, img_size),
    #                                 transforms.ToTensor(),
    #                                 my_normalize
    #                             ]),
    #             ), batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    #         )

    #     return test_loader




