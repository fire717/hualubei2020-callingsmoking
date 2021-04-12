import cv2
import albumentations as A
import os
import sys
import json


def getAllName(file_dir, tail_list = ['.jpg','.png']): 
    L=[] 
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] in tail_list:
                L.append(os.path.join(root, file))
    return L





read_dir = "finalall"
save_dir = "v3_just"


imgs = getAllName(read_dir)
print(len(imgs))

for img_path in imgs:
    
    if "v3" in img_path:
        save_path = img_path.replace(read_dir, save_dir)
        print((img_path, save_path))
        #b
        os.rename(img_path, save_path)