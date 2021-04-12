# -*- coding: utf-8 -*-

import os, sys, glob, argparse
import time
import glob
import json

import numpy as np
import pandas as pd
import cv2
import gc

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


from libs.mAP import get_test_mAP
from libs.tools import getAllName, seed_reproducer, res2item,res2itemClassifyTest,npSoftmax
from libs.model import NetClassify
from libs.data import getDataLoader
from config import cfg










def predict(test_loader, model_list, mode):
    # switch to evaluate mode
    #model.eval()

    res_list = []
    with torch.no_grad():
        #end = time.time()
        for i, (inputs, target, img_names) in enumerate(test_loader):
            print("\r",str(i)+"/"+str(test_loader.__len__()),end="",flush=True)
            #print(img_name,type(inputs))

            # if not isinstance(inputs,list):
            #     inputs_lists = [inputs]
            # else:
            #     inputs_lists = inputs
            inputs = inputs.cuda()

            merge_output = []
            for model in model_list:
                output = model(inputs)
                output = output.data.cpu().numpy().tolist()

                merge_output.append(output)

            
            merge_output = np.array(merge_output)
            #print(tta_output.shape)
            output = np.array([np.sum(merge_output, axis = 0)])/len(model_list)
            output = output[0]

            for i in range(output.shape[0]):

                output_one = output[i][np.newaxis, :]
                output_one = npSoftmax(output_one)
                img_name = os.path.basename(img_names[i])
                item = res2itemClassifyTest(output_one, img_name)

                res_list.append(item)


            

    return res_list



def main(cfg):

    
    print(cfg)
    print("=================================")

    model_name = cfg['model_name']
    img_size = cfg['img_size']
    class_number = cfg['class_number']
    save_dir = cfg['save_dir']
    random_seed = cfg['random_seed']
    mode = ''#cfg['mode']
    test_path = cfg['test_path']
    GPU_ID = cfg['GPU_ID']
    use_TTA = cfg['use_TTA']
    test_batch_size = cfg['test_batch_size']

    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    seed_reproducer(random_seed)


    test_jpg = getAllName(test_path)
    print("total test: ", len(test_jpg))


    kwargs = {'num_workers': 1, 'pin_memory': True}
    device = torch.device("cuda")

    gpu_count = torch.cuda.device_count()
    print("gpu: ", gpu_count)


    model_path_list = glob.glob('./save/merge/%s-%d_*.pth' % (model_name,img_size))
    print(model_path_list)
    model_list = []
    for i,model_path in enumerate(model_path_list):
        model = NetClassify(model_name, class_number).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = torch.nn.DataParallel(model)
        model_list.append(model)

    


        
    input_data = [test_jpg]
    test_loader = getDataLoader("test", input_data,model_name, img_size, test_batch_size, kwargs)

        

    t = time.time()
    res_list = predict(test_loader, model_list, mode)
    print("\n Predict time: ", time.time() - t)




    # save

    save_json_path = "/notebooks/results/hualucup-13183875717.json"
    with open(save_json_path,'w') as f:  
        json.dump(res_list, f, ensure_ascii=False) 


    
    # del model
    # gc.collect()
    # torch.cuda.empty_cache()





if __name__ == '__main__':
    t = time.time()
    print("------------------------")
    print("Start time: ", t)
    argvs = sys.argv
    if len(argvs)==1:
        print("Detect test data A.")
    else:
        cfg['test_path'] = sys.argv[1]

    print("Run path:", cfg['test_path'])
    
    main(cfg)

    t2 = time.time()
    print("End time: ", t2)
    print("Total cost time : ", t2 - t)
    print("------------------------")