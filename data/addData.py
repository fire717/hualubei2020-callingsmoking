import cv2
import albumentations as A
import os
import sys
import json
import random
import numpy as np

random.seed(42)
np.random.seed(42)

def getAllName(file_dir, tail_list = ['.jpg','.png']): #
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L



# To do:
# calling: 1000
# normal:  500
# smoking: 1000
# sc:      2000

fore_img_lists = [[],[],[],[]] #顺序和类别对应
back_img_lists = [[],[],[],[]]


size = 600
# imgs = getAllName('finalall')
# print(len(imgs))
classname_list = [ 'calling', 'normal', 'smoking','smoking_calling']
gen_nums = [500, 100, 500, 1000]

read_dir = 'finalall'
save_dir = 'aug'

# 1. get names
for i, classname in enumerate(classname_list):
    class_apth = os.path.join(read_dir, classname)

    imgs = getAllName(class_apth)#[:100]
    print(classname, len(imgs))

    for img_path in imgs:
        img = cv2.imread(img_path)
        h,w = img.shape[:2]

        if h==size:
            back_img_lists[i].append(img)

        elif h/w>2 and h<550:
            fore_img_lists[i].append(img)

        # if h>size or w>size:
        #     print(img_path)
        #     b

print("finsh get name: ")
# for fore_img_list in fore_img_lists:
#     print(len(fore_img_list))
# for back_img_list in back_img_lists:
#     print(len(back_img_list))
# print('--------')


# 2. paste

for i, classname in enumerate(classname_list):
    print("----start : ", classname)
    fore_img_list = fore_img_lists[i]
    back_img_list = back_img_lists[i]
    if i!=1:
        back_img_list += back_img_lists[1]
    print(len(fore_img_list), len(back_img_list))

    gen_num = gen_nums[i]

    for j in range(gen_num):
        fore_img = random.choice(fore_img_list)
        fore_h ,fore_w = fore_img.shape[:2]

        back_img = random.choice(back_img_list)
        back_img = A.PadIfNeeded(min_height=size, min_width=size, 
                        border_mode=3, p=1)(image=back_img)['image']

        #print("fore: ", fore_img.shape)
        #print("back: ", back_img.shape)

        paste_x = max(1, (size-fore_w)//2 + random.randint(-20,20))
        paste_y = max(1, (size-fore_h)//2 + random.randint(-20,20))
        #print(paste_x, paste_y)

        back_img[paste_y:paste_y+fore_h, paste_x:paste_x+fore_w, :] = fore_img

        save_path = os.path.join(save_dir, classname, "aug_"+str(i)+"_"+str(j)+".jpg")
        cv2.imwrite(save_path, back_img)
        #b