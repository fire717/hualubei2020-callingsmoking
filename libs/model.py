import torch
import torch.nn as nn
import pretrainedmodels
from efficientnet_pytorch import EfficientNet
# from pycls import datasets
# from . import pycls
#from pycls import models as pymodel
import os

# import pycls.core.config as config

# config.load_cfg('configs/regnety/')

# def l2_norm(input, axis=1):
#     norm = torch.norm(input, 2, axis, True)
#     output = torch.div(input, norm)
#     return output
    
# class BinaryHead(nn.Module):
#     def __init__(self, num_class=4, emb_size=2048, s=16.0):
#         super(BinaryHead, self).__init__()
#         self.s = s
#         self.fc = nn.Sequential(nn.Linear(emb_size, num_class))

#     def forward(self, fea):
#         fea = l2_norm(fea)
#         logit = self.fc(fea) * self.s
#         return logit



class NetClassify(nn.Module):
    def __init__(self, model_name, class_number):
        super(NetClassify, self).__init__()

        

        self.name = model_name
                
        ### init model
        if "eff" in model_name:
            #model = EfficientNet.from_name(model_name)
            self.model_feature = EfficientNet.from_name(model_name.replace('adv-',''))

        elif "wsl" in model_name:
            model = torch.hub.load('facebookresearch/WSL-Images', model_name)

        # elif "RegNet" in model_name:

        #     model_cate = model_name.split('-')[-1]
        #     self.model_feature = pymodel.regnety(model_cate, pretrained=False)#, cfg_list=("MODEL.NUM_CLASSES", 4))

        # elif 'EN-B' in model_name:
        #     model_cate = model_name.split('-')[-1]
        #     self.model_feature = pymodel.effnet(model_cate, pretrained=False)
        #     #b

        else:
            #model_name = 'resnext50' # se_resnext50_32x4d xception
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            print(pretrainedmodels.pretrained_settings[model_name])


        # print(model)
        # b
        # print(model_name)
        # b

        ### load model
        if model_name=="resnet50":
            #model.cpu()
            model.load_state_dict(torch.load("../model/resnet50-19c8e357.pth"),strict=False)
            fc_features = model.last_linear.in_features 
        elif model_name=="xception":
            model.load_state_dict(torch.load("../model/xception-43020ad28.pth"),strict=False)
            fc_features = model.last_linear.in_features 
        elif model_name == "se_resnext50_32x4d":
            model.load_state_dict(torch.load("../model/se_resnext50_32x4d-a260b3a4.pth"),strict=False)
            model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_features = model.last_linear.in_features 
        elif model_name == "se_resnext101_32x4d":
            model.load_state_dict(torch.load("../model/se_resnext101_32x4d-3b2fe3d8.pth"),strict=False)
            model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_features = model.last_linear.in_features 
        elif model_name == "resnext101_32x8d_wsl":
            model.load_state_dict(torch.load("../model/ig_resnext101_32x8-c38310e5.pth"),strict=False)
            fc_features = model.fc.in_features 
        elif model_name == "resnext101_32x16d_wsl":
            model.load_state_dict(torch.load("../model/ig_resnext101_32x16-c6f796b0.pth"),strict=False)
            fc_features = model.fc.in_features 

        elif model_name == "efficientnet-b0":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b0-355c32eb.pth"),strict=False) 
        elif model_name == "efficientnet-b1":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b1-f1951068.pth"),strict=False) 
        elif model_name == "efficientnet-b2":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b2-8bb594d6.pth"),strict=False) 
        elif model_name == "efficientnet-b3":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b3-5fb5a3c3.pth"),strict=False)
        elif model_name == "adv-efficientnet-b0":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b0-b64d5a18.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b1":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b1-0f3ce85a.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b2":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b2-6e9d97e5.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b3":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b3-cdd7c0f4.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b4":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b4-44fb3a87.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b5":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b5-86493f6b.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b6":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b6-ac80338e.pth"),strict=False) 
        
        elif model_name == "adv-efficientnet-b7":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b7-4652b6dd.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b8":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b8-22a8fe65.pth"),strict=False) 
        

        elif 'RegNet' in model_name:
            pass
            load_name = model_name+'_dds_8gpu.pyth'
            load_path = os.path.join("../model/dds_baselines/176245422", load_name)
            # # model.load_state_dict(torch.load(load_path),strict=True) 
            checkpoint = torch.load(load_path, map_location="cpu")

            self.model_feature.load_state_dict(checkpoint["model_state"],strict=True)



        elif 'EN-B' in model_name:
            pass
            load_name = model_name+'_dds_8gpu.pyth'
            load_path = os.path.join("../model/dds_baselines/161305098", load_name)
            # # model.load_state_dict(torch.load(load_path),strict=True) 
            checkpoint = torch.load(load_path, map_location="cpu")

            self.model_feature.load_state_dict(checkpoint["model_state"],strict=True)
            

        else:
            raise Exception("Not load pretrained model!")


        if "eff" in model_name:

            #self.model_feature._dropout = nn.Dropout(0.6)

            fc_features = self.model_feature._fc.in_features 
            self.model_feature._fc = nn.Sequential(nn.Linear(fc_features, class_number))
            #self.model_feature._fc = nn.Linear(fc_features, class_number)
            #self.model_feature = nn.Sequential(*list(self.model_feature.children())[:-3])
            # print(self.svm)
            # b
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.25),
                                         # nn.Linear(512, 128), 
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.50), 
                                         # nn.Linear(128,class_number))
            # print(list(self.model_feature.children())[:-3])
            # b
            # self.model_features = nn.Sequential(*list(self.model_feature.children())[:-3])
            # self.last_linear = nn.Linear(fc_features, class_number) 

        elif "RegNet" in model_name:
            # print(model)
            # print(nn.Sequential(*list(model.children())[:-1]))
            # b

            fc_features = self.model_feature.head.fc.in_features 
            self.model_feature.head.fc = nn.Sequential(nn.Linear(fc_features, class_number))

            self.featuremap1 = x.detach()
            # for k,v in self.model_feature.named_parameters():
            #     print('{}: {}'.format(k, v.requires_grad))
            # b

        elif "EN-B" in model_name:
            # print(model)
            # print(nn.Sequential(*list(model.children())[:-1]))
            # b
            fc_features = self.model_feature.head.fc.in_features 
            self.model_feature.head.fc = nn.Sequential(nn.Linear(fc_features, class_number))


        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model_feature = nn.Sequential(*list(model.children())[:-1])
            
            # self.dp_linear = nn.Linear(fc_features, 8) 
            # self.dp = nn.Dropout(0.50)
            self.last_linear = nn.Linear(fc_features, class_number) 
            #self.last_linear = BinaryHead(3, emb_size=2048, s=1)
                
        # print(self.model_feature)
        # b
        
    def forward(self, img):        
        #self.svm = self.svm_feature(img)
        out = self.model_feature(img)
        # out = self.last_linear(out)
        #out = self.avgpool(out)
        if  "RegNet" in self.name or "EN-B" in self.name:
            return out

        if self.name=="xception":
            out = self.avgpool(out)

        if "eff" not in self.name:
            out = out.view(out.size(0), -1)

            
            out = self.last_linear(out)


        return out



class NetClassifySVM(nn.Module):
    def __init__(self, model_name, class_number):
        super(NetClassifySVM, self).__init__()

        self.name = model_name
                
        ### init model
        if "eff" in model_name:
            #model = EfficientNet.from_name(model_name)
            self.model_feature = EfficientNet.from_name(model_name.replace('adv-',''))

        elif "wsl" in model_name:
            model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

        else:
            #model_name = 'resnext50' # se_resnext50_32x4d xception
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            print(pretrainedmodels.pretrained_settings[model_name])


        # print(model)
        # b

        ### load model
        if model_name=="resnet50":
            #model.cpu()
            model.load_state_dict(torch.load("../model/resnet50-19c8e357.pth"),strict=False)
            fc_features = model.last_linear.in_features 
        elif model_name=="xception":
            model.load_state_dict(torch.load("../model/xception-43020ad28.pth"),strict=False)
            fc_features = model.last_linear.in_features 
        elif model_name == "se_resnext50_32x4d":
            model.load_state_dict(torch.load("../model/se_resnext50_32x4d-a260b3a4.pth"),strict=False)
            model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_features = model.last_linear.in_features 
        elif model_name == "se_resnext101_32x4d":
            model.load_state_dict(torch.load("../model/se_resnext101_32x4d-3b2fe3d8.pth"),strict=False)
            model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_features = model.last_linear.in_features 
        elif model_name == "resnext101_32x8d_wsl":
            model.load_state_dict(torch.load("../model/ig_resnext101_32x8-c38310e5.pth"),strict=False)
            fc_features = model.fc.in_features 

        elif model_name == "efficientnet-b0":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b0-355c32eb.pth"),strict=False) 
        elif model_name == "efficientnet-b1":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b1-f1951068.pth"),strict=False) 
        elif model_name == "efficientnet-b2":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b2-8bb594d6.pth"),strict=False) 
        elif model_name == "efficientnet-b3":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b3-5fb5a3c3.pth"),strict=False)
        elif model_name == "adv-efficientnet-b0":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b0-b64d5a18.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b1":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b1-0f3ce85a.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b2":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b2-6e9d97e5.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b3":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b3-cdd7c0f4.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b5":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b5-86493f6b.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b7":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b7-4652b6dd.pth"),strict=False) 



        if "eff" in model_name:
            
            
            #fc_features = self.model_feature._fc.in_features 
            #self.model_feature._fc = nn.Sequential(nn.Linear(fc_features, class_number))
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.25),
                                         # nn.Linear(512, 128), 
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.50), 
                                         # nn.Linear(128,class_number))
            self.model_features = nn.Sequential(*list(self.model_feature.children())[:-3])

        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model_feature = nn.Sequential(*list(model.children())[:-1])
            
            # self.dp_linear = nn.Linear(fc_features, 8) 
            # self.dp = nn.Dropout(0.50)
            self.last_linear = nn.Linear(fc_features, class_number) 
            #self.last_linear = BinaryHead(3, emb_size=2048, s=1)
                
        # print(self.model_feature)
        # b
        
    def forward(self, img):        

        x = self.model_features(img)

        return x




class NetMultilabel(nn.Module):
    def __init__(self, model_name, class_number):
        super(NetMultilabel, self).__init__()
                
        self.name = model_name
                
        ### init model
        if "eff" in model_name:
            #model = EfficientNet.from_name(model_name)
            self.model_feature = EfficientNet.from_name(model_name.replace('adv-',''))

        else:
            #model_name = 'resnext50' # se_resnext50_32x4d xception
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
            print(pretrainedmodels.pretrained_settings[model_name])


        # print(model)
        # b

        ### load model
        if model_name=="resnet50":
            #model.cpu()
            model.load_state_dict(torch.load("../model/resnet50-19c8e357.pth"),strict=False)
            fc_features = model.last_linear.in_features 
        elif model_name=="xception":
            model.load_state_dict(torch.load("../model/xception-43020ad28.pth"),strict=False)
            fc_features = model.last_linear.in_features 
        elif model_name == "se_resnext50_32x4d":
            model.load_state_dict(torch.load("../model/se_resnext50_32x4d-a260b3a4.pth"),strict=False)
            model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_features = model.last_linear.in_features 
        elif model_name == "se_resnext101_32x4d":
            model.load_state_dict(torch.load("../model/se_resnext101_32x4d-3b2fe3d8.pth"),strict=False)
            model.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            fc_features = model.last_linear.in_features 


        elif model_name == "efficientnet-b0":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b0-355c32eb.pth"),strict=False) 
        elif model_name == "efficientnet-b1":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b1-f1951068.pth"),strict=False) 
        elif model_name == "efficientnet-b2":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b2-8bb594d6.pth"),strict=False) 
        elif model_name == "efficientnet-b3":
            self.model_feature.load_state_dict(torch.load("../model/efficientnet-b3-5fb5a3c3.pth"),strict=False)
        elif model_name == "adv-efficientnet-b0":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b0-b64d5a18.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b1":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b1-0f3ce85a.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b2":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b2-6e9d97e5.pth"),strict=False) 
        elif model_name == "adv-efficientnet-b3":
            self.model_feature.load_state_dict(torch.load("../model/adv-efficientnet-b3-cdd7c0f4.pth"),strict=False) 


        if "eff" in model_name:
            # self.model_feature = nn.Sequential(*list(model.children())[:-3])
            fc_features = self.model_feature._fc.in_features 
            self.model_feature._fc = nn.Sequential(nn.Linear(fc_features, class_number))
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.25),
                                         # nn.Linear(512, 128), 
                                         # nn.ReLU(),  
                                         # nn.Dropout(0.50), 
                                         # nn.Linear(128,class_number))

        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.model_feature = nn.Sequential(*list(model.children())[:-1])
            
            # self.dp_linear = nn.Linear(fc_features, 8) 
            # self.dp = nn.Dropout(0.50)
            self.last_linear = nn.Linear(fc_features, class_number) 
            #self.last_linear = BinaryHead(3, emb_size=2048, s=1)
                
        # print(self.model_feature)
        # b
        
    def forward(self, img):        
        out = self.model_feature(img)

        #out = self.avgpool(out)

        if self.name=="xception":
            out = self.avgpool(out)

        if "eff" not in self.name:
            out = out.view(out.size(0), -1)

            # out = self.dp_linear(out)
            # out = self.dp(out)
            
            out = self.last_linear(out)

        return out