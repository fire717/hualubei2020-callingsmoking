# from __future__ import print_function

import os,argparse
import random
import gc
import numpy as np
import cv2
from PIL import Image
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
import torchvision.models as models

from sklearn.model_selection import KFold


from libs.mAP import getValmAP
from libs.tools import *
from libs.model import NetClassify, NetMultilabel
from libs.data import getDataLoader
from libs.mixup import mixup_data, mixup_criterion

from torch.autograd import Variable
from config import cfg

import platform

from libs.scheduler import GradualWarmupScheduler

import glob

#from adabelief_pytorch import AdaBelief

from libs.ranger import Ranger 
from libs.focal_loss import FocalLoss 


def trainClassify(model, 
    device, 
    train_loader, 
    optimizer, 
    epoch, 
    total_epoch,
    criterion,
    use_distill,
    label_smooth):
    
    model.train()
    correct = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data).double()

        #all_linear2_params = torch.cat([x.view(-1) for x in model.model_feature._fc.parameters()])
        #l2_regularization = 0.0003 * torch.norm(all_linear2_params, 2)

        loss = criterion(output, target)# + l2_regularization.item()
        loss.backward() #计算梯度

        clip_gradient(optimizer)

        optimizer.step() #更新参数
        optimizer.zero_grad()#把梯度置零

        ### train acc
        pred_score = nn.Softmax(dim=1)(output)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if use_distill or label_smooth>0:
            target = target.max(1, keepdim=True)[1] 
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += len(data)

        train_acc =  correct / count
        #print(train_acc)
        if batch_idx % 10 == 0:
            print('\r',
                '{}/{} [{}/{} ({:.0f}%)] loss:{:.3f} acc: {:.3f} '.format(
                epoch+1, total_epoch,batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),train_acc), 
                end="",flush=True)


#val_loss: 1.0412, val_acc: 66.67%, val_mAP: 0.6167
def valClassify( model, device, val_loader, criterion, use_distill, label_smooth):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        pres = []
        labels = []
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            #print(target.shape)
            if use_distill:
                output = model(data).double()
            else:
                output = model(data)


            val_loss += criterion(output, target).item() # sum up batch loss

            #print(output.shape)
            pred_score = nn.Softmax(dim=1)(output)
            #print(pred_score.shape)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if use_distill or label_smooth>0:
                target = target.max(1, keepdim=True)[1] 
            correct += pred.eq(target.view_as(pred)).sum().item()


            batch_pred_score = pred_score.data.cpu().numpy().tolist()
            batch_label_score = target.data.cpu().numpy().tolist()
            pres.extend(batch_pred_score)
            labels.extend(batch_label_score)

    pres = np.array(pres)
    labels = np.array(labels)
    #print(pres.shape, labels.shape)


    mAP = getValmAP(pres, labels)

    val_loss /= len(val_loader.dataset)
    val_acc =  correct / len(val_loader.dataset)
    print(' ------------------------------ val_loss: {:.4f}, val_acc: {:.2f}%, val_mAP: {:.4f}'.format(
        val_loss, 100. * val_acc, mAP))

    return val_loss,  mAP





def main(cfg):
    

    print(cfg)
    print("=================================")


    model_name = cfg['model_name']
    img_size = cfg['img_size']
    class_number = cfg['class_number']
    save_dir = cfg['save_dir']
    random_seed = cfg['random_seed']
    train_path = cfg['train_path']
    GPU_ID = cfg['GPU_ID']

    fold_num = cfg['k_flod']
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    learning_rate = cfg['learning_rate']
    early_stop_patient = cfg['early_stop_patient']
    save_start_epoch = cfg['save_start_epoch']
    use_warmup = cfg['use_warmup']
    schedu = cfg['schedu']
    optims = cfg['optims']
    weight_decay = cfg['weight_decay']
    use_distill = cfg['use_distill']
    label_smooth = cfg['label_smooth']
    model_path = cfg['model_path']
    start_fold = cfg['start_fold']

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    seed_reproducer(random_seed)

    ############################################################
    

    # log_interval = 10
    #use_cuda = True
    device = torch.device("cuda")#cuda

    if platform.system() == "Windows":
        kwargs = {'num_workers': 0, 'pin_memory': True}
    else:
        kwargs = {'num_workers': 4, 'pin_memory': True}

    train_names = getAllName(train_path)
    print("total imgs: ", len(train_names))

    if not use_distill:
        train_names = [x for x in train_names if "aug" not in x]
        print("remove aug: ", len(train_names))

    #print(train_names[:3])
    train_names.sort(key = lambda x:os.path.basename(x))
    #print(train_names[:3])

    train_names = np.array(train_names)
    random.shuffle(train_names)

 

    
    folds = KFold(n_splits=fold_num, shuffle=False)#, random_state=random_seed
    for fold_i, (train_index, val_index) in enumerate(folds.split(train_names)):
        print("Fold: ", fold_i+1,'/',fold_num)
        if fold_i<start_fold:
            continue


        train_data = train_names[train_index]
        val_data = train_names[val_index]
        # print(val_data[-3:])
        # b
        input_data = [train_data, val_data]
        


        if not use_distill and label_smooth==0:
            criterion = torch.nn.CrossEntropyLoss().cuda()
            #criterion = FocalLoss().cuda()
            train_loader, val_loader = getDataLoader("trainClassify", input_data,model_name, img_size, batch_size, kwargs)
        else:

            criterion = CrossEntropyLossOneHot().cuda()
            #kwargs['use_distill'] = use_distill
            #print(kwargs)
            train_loader, val_loader = getDataLoader("trainClassifyOnehot", input_data,model_name, img_size, batch_size, kwargs)


        model = NetClassify(model_name, class_number).to(device)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
            print("---------------------- load model!!!")
        # print(model)
        # b
        

        if optims=='adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optims=='SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif optims=='AdaBelief':
            optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-12, betas=(0.9,0.999))
        elif optims=='Ranger':
            optimizer = Ranger(model.parameters(), lr=learning_rate)
        


        if schedu=='default':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        elif schedu=='step1':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8, last_epoch=-1)
        elif schedu=='step2':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5, last_epoch=-1)
        elif schedu=='step3':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, last_epoch=-1)
        elif schedu=='SGDR1': 
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=10, 
                                                                T_mult=2)
        elif schedu=='SGDR2': 
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                 T_0=5, 
                                                                T_mult=2)

        # elif schedu=='CVPR': 
        #     scheduler = WarmRe   mizer, T_max=10, T_mult=1, eta_min=1e-5)
        
        if use_warmup:
            scheduler_warmup = GradualWarmupScheduler(optimizer, 
                multiplier=1, total_epoch=1, after_scheduler=scheduler)





        early_stop_value = 0
        early_stop_dist = 0

        for epoch in range(epochs):
            
            if schedu=='step3':
                if epoch==10:
                    img_size=416
                    batch_size = 4
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, last_epoch=-1)
                    if not use_distill:
                        criterion = torch.nn.CrossEntropyLoss().cuda()
                        train_loader, val_loader = getDataLoader("trainClassify", input_data,model_name, img_size, batch_size, kwargs)
                    else:
                        criterion = CrossEntropyLossOneHot().cuda()
                        train_loader, val_loader = getDataLoader("trainClassifyOnehot", input_data,model_name, img_size, batch_size, kwargs)


                elif epoch==15:
                    img_size=600
                    batch_size = 3
                    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10, 
                                                                    T_mult=2)
                    if not use_distill:
                        criterion = torch.nn.CrossEntropyLoss().cuda()
                        train_loader, val_loader = getDataLoader("trainClassify", input_data,model_name, img_size, batch_size, kwargs)
                    else:
                        criterion = CrossEntropyLossOneHot().cuda()
                        train_loader, val_loader = getDataLoader("trainClassifyOnehot", input_data,model_name, img_size, batch_size, kwargs)


            

            trainClassify(model, device, train_loader, optimizer, epoch, epochs, criterion, use_distill, label_smooth)
            print(" LR:", optimizer.param_groups[0]["lr"], end="")

            t = time.time()
            val_loss, mAP = valClassify(model, device, val_loader, criterion, use_distill, label_smooth)
            print("val time: ", time.time() - t)

            #print('333')
            #continue

            if use_warmup:
                scheduler_warmup.step(epoch)
            else:
                if schedu=='default':
                    scheduler.step(mAP)
                else:
                    scheduler.step()


            #print("---")
            #print(mAP, early_stop_value, early_stop_dist)
            if mAP>early_stop_value:
                early_stop_value = mAP
                early_stop_dist = 0
                if epoch>=save_start_epoch:
                    hitory_path = glob.glob('./save/%s-%d_*k-%d_%s.pth' % (model_name,img_size,fold_i,GPU_ID))
                    if len(hitory_path)!=0:
                        if os.path.exists(hitory_path[0]):
                            os.remove(hitory_path[0])
                    torch.save(model.state_dict(), './save/%s-%d_%d_%.4f_k-%d_%s.pth' % (model_name,img_size,epoch,mAP,fold_i,GPU_ID))
            



            early_stop_dist+=1
            if early_stop_dist>early_stop_patient:
                print("------")
                print(cfg)
                print("------")
                print("===== Early Stop with patient %d , best is Epoch - %d :%f" % (early_stop_patient,epoch-early_stop_patient,early_stop_value))
                break
            if  epoch+1==epochs:
                print("===== Finish trainging , best is Epoch - %d :%f" % (epoch-early_stop_dist,early_stop_value))
                break

            


                
        del model
        gc.collect()
        torch.cuda.empty_cache()


        #if not use_distill:
        #    break
        #break


if __name__ == '__main__':
    main(cfg)