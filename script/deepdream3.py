# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

#!wget -nc https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip -n inception5h.zip

#python3
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
from io import BytesIO
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf


#input_image = raw_input('input image:>>')
input_image = 's_cat'
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

count = []
layer_name = ''
def showarray(a ,i=0):
    count.append(i)
    a = np.uint8(np.clip(a, 0, 1)*255)
    s = PIL.Image.fromarray(a)
    s.save('/home/roboworks/deepdream/image/{}/{}_{}.jpg'.format(input_image,layer_name,len(count)))

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
            print('----- x:{} -----'.format(x))
            sub = img_shift[y:y+sz,x:x+sz]
            g,score,obj = sess.run([t_grad, t_score, t_obj], {t_input:sub})
            print('obj:','shape:',obj.shape)
            print('Sc:',score)
            print('grad:',g[0][0],g.shape)
            print('I:',img[0][0])
            grad[y:y+sz,x:x+sz] = g
        #print('grad:',g[0][0],g.shape)
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

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
        print('-------------------octave {}------------------'.format(octave+1))
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

#input_image = 'kuro'
img0 = np.float32(img0)
print('input image shape:',img0.shape)
#layer_name = 'conv2d0'
layer_name = 'mixed4c'
#layer_name = 'maxpool0'
#render_deepdream(T(layer_name), img0)
#render_deepdream(tf.square(T(layer_name)), img0)
render_deepdream(tf.square(T(layer_name)), img0, iter_n=20, octave_n=4)

