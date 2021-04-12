import os
import random
import numpy as np

import torch
import torch.nn as nn

def getAllName(file_dir, tail_list = ['.png','.jpg','.JPG','.PNG']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L

def npSoftmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def seed_reproducer(seed=42):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def res2itemClassify(res, target):
    #print(res)#[[0.6018679 0.2981321, 0.1]]
    cate_name = {0:"calling", 1:"normal", 2:"smoking", 3:"smoking_calling"}
    scores = res[0]

    category = cate_name[np.argmax(scores)]
    score = round(float(np.max(scores)), 5)
    item = {"image_name":  img_name, 
                "category": category, 
                "score": score}
    return item

def res2itemClassifyTest(res, img_name):
    #print(res)#[[0.6018679 0.3981321]]
    cate_name = {0:"calling", 1:"normal", 2:"smoking", 3:"smoking_calling"}
    # cate_name = {0:"calling", 1:"smoking"}
    scores = res[0]

    category = cate_name[np.argmax(scores)]
    score = round(float(np.max(scores)), 5)
    item = {"image_name":  img_name, 
                "category": category, 
                "score": score}
    return item

def res2item(res, img_name):
    #print(res)#[[0.6018679 0.3981321]]
    #cate_name = {0:"calling", 1:"normal", 2:"smoking"}
    cate_name = {0:"calling", 1:"smoking"}
    scores = res[0]
    #print(scores)
    if np.max(scores) > 0.5:
        score = round(float(np.max(scores)), 5)
        category = cate_name[np.argmax(scores)]
    else:
        #score = round(float((1.0*2 - sum(scores))/2.0), 5)
        score = round(float(1.0 - max(scores)), 5)
        category = "normal"
    # category = cate_name[np.argmax(scores)]
    # score = round(float(np.max(scores)), 5)
    item = {"image_name":  img_name, 
                "category": category, 
                "score": score}
    return item




def clip_gradient(optimizer, grad_clip=1):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def transferMutilToClass(datalist):
    # n*2 [call,smoke] -> n*3
    new_list = []
    for data in datalist:
        # if max(scores) > 0.5:
        #     score = max(scores)
        #     if np.argmax(scores)==0:
        #         [data[0], min(1-data[0],1-data[1]), data[1]]
        # else:
        #     #score = round(float((1.0*2 - sum(scores))/2.0), 5)
        #     score = round(float(1.0 - max(scores)), 5)
        #     category = "normal"

        #print(data)
        new_data = [data[0]+1-data[1], 
                    1-data[0]+1-data[1], 
                    data[1]+1-data[0], 
                    data[0]+data[1]] #call normal smoke sc
        # new_data = np.array(new_data)
        # new_data /= sum(new_data)
        # print(new_data)
        new_data = npSoftmax(np.array(new_data)).tolist()
        # print(new_data)
        # b
        new_list.append(new_data)
    return new_list

def transferMutilLabel(datalist):
    # n*2 [call,smoke] -> n*1
    new_list = []
    for data in datalist:
        if data[0]<0.5:
            if data[1]<0.5:
                new_data = 1
            else:
                new_data = 2
        else:
            if data[1]<0.5:
                new_data = 0
            else:
                new_data = 3
        new_list.append(new_data)
    return new_list

class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))

        
if __name__ == '__main__':
    # x = torch.FloatTensor([1.0399,0.1582])
    # print(nn.Sigmoid()(x))

    # x = torch.FloatTensor([0,1])
    # print(nn.Sigmoid()(x))

    x = [[0.0030388175509870052, 0.002976007293909788], 
        [1.3112669876136351e-05, 0.9992826581001282],
        [0.5, 0.9992826581001282]]
    print(transferMutilToClass(x))