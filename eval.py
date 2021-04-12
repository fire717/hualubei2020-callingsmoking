# -*- coding: utf-8 -*-

import os, sys, glob, argparse
import time
import glob
import json
import random
import numpy as np
import pandas as pd
import cv2
# from PIL import Image
import gc

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data.dataset import Dataset
# import torchvision.transforms.functional as F
# import pretrainedmodels
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold


from libs.mAP import getValmAP
from libs.tools import getAllName, seed_reproducer, res2item,res2itemClassifyTest,npSoftmax
from libs.model import NetClassify
from libs.data import getDataLoader
from config import cfg










def predict(test_loader, model, mode, device):
    # switch to evaluate mode
    model.eval()

    res_list = []
    with torch.no_grad():
        #end = time.time()
        pres = []
        labels = []
        for i, (data, target, img_name) in enumerate(test_loader):
            print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)
            #print(img_name,type(inputs))
            data, target = data.to(device), target.to(device)
            output = model(data)


            #val_loss += criterion(output, target).item() # sum up batch loss

            # print(output)
            pred_score = nn.Softmax(dim=1)(output)
            # print(pred_score)
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()


            batch_pred_score = pred_score.data.cpu().numpy().tolist()
            batch_label_score = target.data.cpu().numpy().tolist()
            pres.extend(batch_pred_score)
            labels.extend(batch_label_score)

    pres = np.array(pres)
    labels = np.array(labels)
    #print(pres.shape, labels.shape)


    mAP = getValmAP(pres, labels)
        #test_pred = np.vstack(test_pred)
    return mAP



def main(cfg ):

    
    print(cfg)
    print("=================================")

    model_name = cfg['model_name']
    img_size = cfg['img_size']
    class_number = cfg['class_number']
    save_dir = cfg['save_dir']
    random_seed = cfg['random_seed']
    mode = cfg['mode']
    train_path = cfg['train_path']
    GPU_ID = cfg['GPU_ID']

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    seed_reproducer(random_seed)

    device = torch.device("cuda")
    # test_jpg = getAllName(test_path)
    # print("total test: ", len(test_jpg))

    label_json_path = '../data/test_clean.json'
    kwargs = {'num_workers': 1, 'pin_memory': True}
    device = torch.device("cuda")



    
    train_names = getAllName(train_path)
    print("total imgs: ", len(train_names))
    if mode != 'mutillabel':
        train_names = [x for x in train_names if "smoking_calling_images" not in x]
        print("remove smoking_calling: ", len(train_names))
    #print(train_names[:3])
    train_names.sort(key = lambda x:os.path.basename(x))
    #print(train_names[:3])
    train_names = np.array(train_names)
    random.shuffle(train_names)
    folds = KFold(n_splits=5, shuffle=False)#, random_state=random_seed


    for fold_i, (train_index, val_index) in enumerate(folds.split(train_names)):
        print("Fold: ", fold_i)
        model_path_list = glob.glob('./save/%s-%d_*-0_0.pth' % (model_name,img_size))
        model_path = model_path_list[0]
        print(model_path)
        
        val_data = train_names[val_index]
        # print(len(val_data))
        # print(val_data[-20:])
        # b
        input_data = [val_data]
        test_loader = getDataLoader("test", input_data,model_name, img_size, 1, kwargs)

            
         # "cpu")cuda

        if mode=="mutillabel":
            model = NetClassify(model_name, class_number).to(device)
        else:
            model = NetClassify(model_name, class_number).to(device)
        model.load_state_dict(torch.load(model_path))
        # print(model)
        # b
        # model = nn.DataParallel(model).cuda()
        t = time.time()
        mAP = predict(test_loader, model, mode, device)
        print("time: ", time.time() - t)
        print("mAP: ",mAP)
        #print(test_pred[0])


        # save_json_path = "save/result_val.json"
        # with open(save_json_path,'w') as f:  
        #     json.dump(res_list, f, ensure_ascii=False) 



        # classname_list = ['smoking', 'calling', 'normal']
        # mAP = get_test_mAP(save_json_path, classname_list, label_json_path)
        # print("mAP: ", mAP)

        # model_mAP_list.append([os.path.basename(model_path), mAP])
        # print("--------------------------")

        del model
        gc.collect()
        torch.cuda.empty_cache()

        break

    # model_mAP_list.sort(key=lambda x:x[1], reverse=True)
    # print("=================================")
    # print(model_mAP_list[:3])



if __name__ == '__main__':
    main(cfg)