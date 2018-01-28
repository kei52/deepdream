# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

#!wget -nc https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip -n inception5h.zip

#python3
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import math
from io import BytesIO
import numpy as np
from functools import partial
import PIL
from PIL import Image
import tensorflow as tf

#sys.stdout = open("tmep.txt","w")
#input_image = raw_input('save name:>>')
input_image = ''
model_fn = 'tensorflow_inception_graph.pb'
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
#expand_dims(image, 0)これにより、形が作成され[1, height, width, channels]
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]

layer_list = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
total_layer = len(layer_list)
add_layer = []
for i in layer_list:
    str_layer = (str(i).replace('import/',''))
    str_layer = (str_layer.replace('/conv',''))
    add_layer.append(str_layer)

img_noise = np.random.uniform(size=(224,224,3)) + 100.0
count = []
def showarray(a, fmt='jpeg' ,i=0):
    print('----------showarray------------------')
    count.append(i)
    #print(a[0][0],np.clip(a, 0, 1)[0][0],(a*255)[0][0],np.uint8(np.clip(a, 0, 1)*255)[0][0])
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    s = PIL.Image.fromarray(a)
    s.save('/home/roboworks/deepdream/image/render_naive/{}.jpg'.format(input_image))
    print('\n')

def visstd(a, s=0.1):
    #可視化するための画像範囲を正規化
    print('----------------visstd---------------')
    print('a.mean():{}'.format(a.mean()),'max(a.std():{}, 1e-4):{}'.format(max(a.std(), 1e-4),a.std()),'')
    print('a-a.mean())/max(a.std(), 1e-4)*s + 0.5:',((a-a.mean())/max(a.std(), 1e-4)*s + 0.5)[0][0])
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

# get layer func
def T(layer):
    '''Helper for getting layer output tensor'''
    print('function T(layer):',graph.get_tensor_by_name("import/%s:0"%layer))
    return graph.get_tensor_by_name("import/%s:0"%layer)

# use t_inputdata
# 層にわかりやすく反応させるため（値が似ているため）にグレー画像からrgbを計算していく
def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    #print('----------render naive----------')
    #print('input_obj_name:',t_obj.shape,type(t_obj))
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    #t_scoreをt_inputで微分を行う
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    img = img0.copy()#imgの初期化
    for i in range(iter_n):#20回同様の作業を行う
        #gには[t_grad, t_score],scoreには [t_input:img]
        print('----------------number {}---------------'.format(i+1))
        print('t_input:',img[0][0])
        score, g, obj = sess.run([t_score, t_grad, t_obj], {t_input:img})
        print('obj',obj[0][0][0][:3],obj.shape)
        print('Sc:',score)
        print('grad:',g[0][0])
        print('grad.std()',g.std())
        print('grad / grad.std:{}'.format((g/g.std())[0][0]),'+1e-8')
        g = g / g.std() + 1e-8
        print('grad_:',g[0][0])
        print('I:{} + grad:{}'.format(img[0][0],(g*step)[0][0]))
        img = img + g*step
        print('I_:',img[0][0])
    print('final I:',img[0][0])
    showarray(visstd(img))
#139
channel = 139 # picking some feature channel to visualize

#layer = 'mixed4d_3x3_bottleneck_pre_relu'
layer = 'conv2d2_pre_relu'
#layer = 'mixed4a_1x1_pre_relu'

input_image = raw_input('input image:>>')
img0 = PIL.Image.open('/home/roboworks/deepdream/image/{}.jpg'.format(input_image))
img0 = np.float32(img0)
print('img0:',img0[0][0])


n = raw_input('iter_n:>>')
n = int(n)
#render_naive(T(layer)[:,:,:,channel],img0=img0,iter_n=n)
layer_name = 'conv2d0'
#layer_name = 'maxpool0'
#layer_name = 'mixed4c'

render_naive(tf.square(T(layer_name)),img0=img0,iter_n=n)

