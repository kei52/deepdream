import numpy as np
import time
import os
import sys
from PIL import Image

def showarray(img,s_name):
    showarray = np.uint8(img)
    print 'showarray:',showarray[0][0]
    pil_img = Image.fromarray(showarray)
    pil_img.save('/home/roboworks/Pictures/{}.jpg'.format(s_name))             

def pyramid(img,iter_n=1):
    for i in range(iter_n):
        print '-----------number {}-----------'.format(i+1)
        out_img = np.copy(img)
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                out_img = img[x][y] / 2
        #showarray(out_img)
    return out_img

def half(img):
    half_img = img/2
    showarray(half_img,'half_img')
    return half_img

def r_only(img):
    r_img = np.copy(img)
    print 'r_img:',r_img.shape
    return r_img

if __name__ == '__main__':
    input_img = Image.open('/home/roboworks/Pictures/dog.jpg')
    input_img = np.float32(input_img)
    print 'input img size:{}'.format(input_img.shape)
    r_only(input_img)
    #pyramid(img=input_img,iter_n=3)
    #half(half(half(input_img)))


