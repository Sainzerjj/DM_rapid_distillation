import torch
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
def addTran(img, attn, factor=0.7):
    img = img.convert('RGBA')
    attn_list = list(attn.getdata())
    print(attn_list[10000:10200])
    img_list = list(img.getdata())
    print(type(img_list))
    # img = Image.new('RGBA', img.size, (255,0,0,255))
    img_blender = Image.new('RGBA', img.size, (0,0,255,255))
    img_blender_list = list(img_blender.getdata())
    
    for i in range(len(img_list)):
        # print(type(img_list[i]))
        if attn_list[i]==(255, 255, 255):
            a = torch.Tensor(list(img_list[i]))
            b = torch.Tensor(list(img_blender_list[i]))
            a[:3] = (a[:3]*factor + b[:3]*(1-factor)).int()
            a = a.tolist()
            a = list(map(int, a))
            # print(type(a))
            a =tuple(a)
            img_list[i] = a
    # img_list = list(map(int, img_list))
    # print(type(img_list))
    img.putdata(img_list)
    # img = 0.3 * img_blender + 0.7 * img
    # Image.blend(img_blender, img, factor)
    return img
img = Image.open("result_pred_x.png")
attn = Image.open("10.png")
# trans = transforms.ToTensor()
# print(img)
# img = trans(img)
img=addTran(img, attn, 0.4)
img.save("result_pred_x_over.png")


