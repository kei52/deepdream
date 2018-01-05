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
#ipython
#from IPython.display import clear_output, Image, display, HTML
import tensorflow as tf
import matplotlib.pyplot as plt
import time

#sys.stdout = open("tmep.txt","w")
#input_image = raw_input('save name:>>')
input_image = ''
model_fn = 'tensorflow_inception_graph.pb'
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
# load pb file
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# t_input is serching where layer
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
# imagenet_mean?
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

# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'mixed4d_3x3_bottleneck_pre_relu'
#139
channel = 139 # picking some feature channel to visualize
# 最初にグレイの画像を生成し１から０のランダムな値（ノイズ）を加える
img_noise = np.random.uniform(size=(224,224,3)) + 100.0
count = []
def showarray(a, fmt='jpeg' ,i=0):
    print('----------showarray----------')
    #print('showarray a.shape:',a.shape,type(a))
    count.append(i)
    a = np.uint8(np.clip(a, 0, 1)*255)
    print(a[0][0])
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    s = PIL.Image.fromarray(a)
    s.save('/home/roboworks/deepdream/image/render_naive/{}.jpg'.format(input_image))

def visstd(a, s=0.1):
    #可視化するための画像範囲を正規化
    print('----------------visstd---------------')
    print(((a-a.mean())/max(a.std(), 1e-4)*s + 0.5)[0][0])
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

# get layer func
def T(layer):
    '''Helper for getting layer output tensor'''
    print('function T(layer):',graph.get_tensor_by_name("import/%s:0"%layer))
    return graph.get_tensor_by_name("import/%s:0"%layer)

# use t_inputdata
# 層にわかりやすく反応させるため（値が似ているため）にグレー画像からrgbを計算していく
def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
    print('----------render naive----------')
    print('input_obj_name:',t_obj.shape,type(t_obj))
    t_score = tf.reduce_mean(t_obj) # defining the optimization objectiveオプティマイズする＝微分して引く更新
    #t_scoreをt_inputで微分を行う
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!＝微分gradient
    img = img0.copy()#img0同じ配列を二つ用意し別々に扱うため?
    for i in range(iter_n):#20回同様の作業を行う
        #gには[t_grad, t_score],scoreには [t_input:img]
        print('----------------number {}---------------'.format(i+1))
        print('t_input:',img[0][0],type(img))
        score, g = sess.run([t_score, t_grad], {t_input:img})#g, scoreにRUN（処理）させて出した値が入る
        print('gradient answer:',g[0][0])
        print('score:',score,type(t_score))
        # normalizing the gradient, so the same step size should work
        # グラデーションを正規化するので、同じステップサイズで作業する必要があります。
        #print('g:',g)極小の値,stdはgの標準偏差を求めるgの値が小さくなるように処理していく
        print('g.std()',g.std())
        # 1e-8=0.00000001
        #for different layers and networks g/g.std()+10^-8
        g = g / g.std()+1e-8
        img = img + g*step
    print('img:',img[0][0])
    showarray(visstd(img))

#print('T(layer)[:,:,:,channel]:',T(layer)[:,:,:,channel])
n = raw_input('iter_n:>>')
n = int(n)
#render_naive(T(layer)[:,:,:,channel],iter_n=n)
from common import common_layer
for i in common_layer.LAYER_LIST:
    print(i)
    input_image = i + '_' + str(n)
    render_naive(T(i)[:,:,:,channel],iter_n=n)
sys.exit()
