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
import matplotlib.pyplot as plt
import time
#input_image = raw_input('input image:>>')
input_image = 'asagao'
if os.path.exists('/home/roboworks/deepdream/image/{}'.format(input_image)) == False:
    os.mkdir('/home/roboworks/deepdream/image/{}'.format(input_image))
img0 = PIL.Image.open('/home/roboworks/deepdream/image/{}.jpg'.format(input_image))

model_fn = 'tensorflow_inception_graph.pb'
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

def strip_consts(graph_def, max_const_size=32):
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
def show_graph(graph_def, max_const_size=32):
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))

tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
show_graph(tmp_def)

channel = 139

count = []
layer_name = ''
def showarray(a, fmt='jpeg' ,i=0):
    count.append(i)
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    s = PIL.Image.fromarray(a)
    s.save('/home/roboworks/deepdream/image/{}/{}.jpg'.format(input_image,layer_name))

def T(layer):
    return graph.get_tensor_by_name("import/%s:0"%layer)

def tffunc(*argtypes):
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]  
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, t_score, t_obj, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            print('x:',x)
            sub = img_shift[y:y+sz,x:x+sz]
            g,score,obj = sess.run([t_grad, t_score, t_obj], {t_input:sub})
            print('obj:',obj[0][0][0][0],obj.shape)
            print('grad:',g[0][0],g.shape)
            print('Sc:',score)
            print('I:',img[0][0])
            grad[y:y+sz,x:x+sz] = g
        #print('grad:',g[0][0],g.shape)
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

lap_graph = tf.Graph()
with lap_graph.as_default():
    lap_in = tf.placeholder(np.float32, name='lap_in')
    lap_out = lap_normalize(lap_in)
show_graph(lap_graph)

img_noise = np.random.uniform(size=(224,224,3)) + 100.0

def render_deepdream(t_obj,img0=img_noise,iter_n=20, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    for octave in range(octave_n):
        print('----------------octave {}---------------'.format(octave+1))
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            print('----------------number {}---------------'.format(i+1))
            print('I:',img0[0][0])
            g = calc_grad_tiled(img, t_grad, t_score, t_obj)
            img = img + g*(step / (np.abs(g).mean()+1e-7))
            print('img = img + g*(step / (np.abs(g) {}.mean() {}+1e-7))'.format(np.abs(g)[0][0],np.abs(g).mean()))
            print('g*(step / (np.abs(g).mean()+1e-7)):',(g*(step / (np.abs(g).mean()+1e-7)))[0][0])
            print('I_:',img[0][0])
            print('showarray:',(img/255.0)[0][0])
            print('\n')
        showarray(img/255.0)

img0 = np.float32(img0)
print('input image:',img0.shape)
layer_name = 'conv2d0'
#render_deepdream(tf.square(T(layer_name)), img0)
#layer_name = 'maxpool0'
#render_deepdream(tf.square(T(layer_name)), img0)
render_deepdream(tf.square(T(layer_name)))
#render_deepdream(tf.square(T(layer_name)), img0, iter_n=20, octave_n=4)

