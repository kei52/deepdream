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

# ready input image
input_image = raw_input('input image:>>')
if os.path.exists('/home/roboworks/deepdream/image/{}'.format(input_image)) == False:
    os.mkdir('/home/roboworks/deepdream/image/{}'.format(input_image))
img0 = PIL.Image.open('/home/roboworks/deepdream/image/{}.jpg'.format(input_image))

model_fn = 'tensorflow_inception_graph.pb'
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
#print('graph',tf.gfile.FastGFile(model_fn,'rb'))
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
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
#print('Number of layers', len(layers))
#print('Total number of feature channels:', sum(feature_nums))

# TFグラフの視覚化をする為に使う関数(strip_consts rename_nodes show_graph)の準備
# Helper functions for TF Graph visualization
# consts = block scoop
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    # loading pb file
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
# グラフの可視化を行う関数 strip_constsを扱う
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
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
    #display(HTML(iframe))
    #print('HTML(iframe)',HTML(iframe))
# Visualizing the network graph. Be sure expand the "mixed" nodes to see their 
# internal structure. We are going to visualize "Conv2D" nodes.
# visualize conv2d
tmp_def = rename_nodes(graph_def, lambda s:"/".join(s.split('_',1)))
#グラフの可視化を行う
show_graph(tmp_def)

#==================================================================

# Picking some internal layer. Note that we use outputs before applying the ReLU nonlinearity
# to have non-zero gradients for features with negative initial activations.
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139 # picking some feature channel to visualize

# 最初にグレイの画像を生成し１から０のランダムな値（ノイズ）を加える
img_noise = np.random.uniform(size=(224,224,3)) + 100.0

# 画像の値を表示する為に変換
count = []
layer_name = ''
test_name = 'faile'

def showarray(a, fmt='jpeg' ,i=0):
    print('----------showarray----------')
    #print('showarray a.shape:',a.shape,type(a))
    count.append(i)
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    s = PIL.Image.fromarray(a)
    if len(count) > 3 and len(count)%4 == 0:
        s.save('/home/roboworks/deepdream/image/{}/{}.jpg'.format(input_image,layer_name))
    else:
        s.save('/home/roboworks/deepdream/image/etc/{}.jpg'.format(test_name))
    #display(Image(data=f.getvalue()))



def visstd(a, s=0.1):
    #可視化するための画像範囲を正規化
    '''
    print(a,'\n',a.shape)
    print(a.mean())
    print(a.std())
    print(max(a.std(), 1e-4))
    print((a-a.mean())/max(a.std(), 1e-4)*s + 0.5)
    '''
    #print('tttttttttttttt',np.amax((a-a.mean())/max(a.std(), 1e-4)*s + 0.5))
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

# get layer func
def T(layer):
    '''Helper for getting layer output tensor'''
    print('function T(layer):',graph.get_tensor_by_name("import/%s:0"%layer))
    return graph.get_tensor_by_name("import/%s:0"%layer)

# use t_inputdata
# 層にわかりやすく反応させるため（値が似ているため）にグレー画像からrgbを計算していく
def render_naive(t_obj, img0=img_noise, iter_n=20, step=3.0):
    print('----------render naive----------')
    print('input_obj_name:',t_obj.shape,type(t_obj))
    t_score = tf.reduce_mean(t_obj) # defining the optimization objectiveオプティマイズする＝微分して引く更新
    #t_scoreをt_inputで微分を行う
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!＝微分gradient
    img = img0.copy()#img0同じ配列を二つ用意し別々に扱うため?
    for i in range(iter_n):#20回同様の作業を行う
        #gには[t_grad, t_score],scoreには [t_input:img]
        #print('t_score:',t_score)
        #print('t_grad:',t_grad)
        #微分とスコアを走らせる？
        print('t_grad:',t_grad,type(t_grad))
        print('t_score:',t_score,type(t_score))
        print('t_input:',img,type(img))
        g, score = sess.run([t_grad, t_score], {t_input:img})#g, scoreにRUN（処理）させて出した値が入る
        print('g:',g)
        print('score:',score,type(t_score))
        # normalizing the gradient, so the same step size should work
        #print('g:',g)極小の値,stdはgの標準偏差を求めるgの値が小さくなるように処理していく
        print('g.std()',g.std())
        g /= g.std()+1e-8         #for different layers and networks g/g.std()+10^-8
        img += g*step
        #print(score, end = ' ')
    #clear_output()
    #print('img:',img.shape)
    time.sleep(5)
    print('visstd(img):',visstd(img).shape)
    showarray(visstd(img))

#t_objにT(layer)[:,:,:,channel]
print('T(layer)[:,:,:,channel]:',T(layer)[:,:,:,channel])
test_name = raw_input('input test name:>>')
render_naive(T(layer)[:,:,:,channel])
#sys.exit()





# f
def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        #print('out:',out)
        def wrapper(*args, **kw):
            #print('*args:',args)
            #print('*kw:',kw)
            #print('tffunc',out.eval(dict(zip(placeholders, args)), session=kw.get('session')))
            #print('tffunc.shape',out.eval(dict(zip(placeholders, args)), session=kw.get('session')).shape)
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap\

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    #print('resize_img:',img)
    #print('resize_return',tf.image.resize_bilinear(img, size)[0,:,:,:])
    return tf.image.resize_bilinear(img, size)[0,:,:,:]  
resize = tffunc(np.float32, np.int32)(resize)


#print('resize',resize)





def calc_grad_tiled(img, t_grad, tile_size=512):
    #print('----------calc grad tiled----------')
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    #print('img:',img.shape)
    #sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(tile_size, size=2)
    #numpy.roll(a, shift, axis=None)
    #配列の要素を回転させる
    #print(sx)
    #print(np.roll(img,sx,1))
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    #print('img_shift:',img_shift)
    for y in range(0, max(h-tile_size//2, tile_size),tile_size):
        for x in range(0, max(w-tile_size//2, tile_size),tile_size):
            sub = img_shift[y:y+tile_size,x:x+tile_size]
            #print('sub:',sub,'\ntype:',type(sub),'\nlen',len(sub))
            g = sess.run(t_grad, {t_input:sub})
            #print('t_grad:',t_grad)
            #print('sub:',sub)
            grad[y:y+tile_size,x:x+tile_size] = g
            #print('sub:',sub.shape)
            #print('g=grad[y:y+sz,x:x+sz]:',g[0][0])#512 512 3
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)



def render_multiscale(t_obj, img0=img_noise, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    print('----------render multiscale----------')
    print('t_obj:',t_obj)
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    # t_scoreをt_inputで微分する
    print('t_score:',t_score)
    print('t_input:',t_input)
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    print('t_grad:',t_grad)
    #print('input_img_render_multiscale',img0.shape)
    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8 # for different layers and networks
            img += g*step
            #print('.', end = ' ')
        #clear_output()
        showarray(visstd(img))
    print('finish render multiscale')

# 複数のパネルごとに処理をおこなうらしい
#print('T(layer)[:,:,:,channel]',T(layer)[:,:,:,channel])
#render_multiscale(T(layer)[:,:,:,channel])

#sys.exit()

# k = kernel? hight pass filter
k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

# Showing the lap_normalize graph with TensorBoard
lap_graph = tf.Graph()
with lap_graph.as_default():
    lap_in = tf.placeholder(np.float32, name='lap_in')
    lap_out = lap_normalize(lap_in)
show_graph(lap_graph)

# lapnormを与える
def render_lapnorm(t_obj, img0=img_noise, visfunc=visstd,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, scale_n=lap_n))

    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            g = lap_norm_func(g)
            img += g*step
            #print('.', end = ' ')
        #clear_output()
        showarray(visfunc(img))

'''
render_lapnorm(T(layer)[:,:,:,channel])


render_lapnorm(T(layer)[:,:,:,65])

render_lapnorm(T('mixed3b_1x1_pre_relu')[:,:,:,101])

render_lapnorm(T(layer)[:,:,:,65]+T(layer)[:,:,:,139], octave_n=4)
'''




# arg layer_name
def render_deepdream(t_obj,img0=img_noise,iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
        #print('octaves',type(octaves))#-0.2,
    
    # generate details octave by octave
    # 4 
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g*(step / (np.abs(g).mean()+1e-7))
            #print('.',end = ' ')
        #clear_output()
        showarray(img/255.0)

#sys.exit()
img0 = np.float32(img0)
#print (img0)
#showarray(img0/255.0)

from common import layer_dict
#render_deepdream(T(layer)[:,:,:,1], img0)

# deepdream ==> input original image
# can not input ==> ['softmax1','nn1','nn0','output1','head1','head0','output','softmax0','input']

'''
for i in layer_dict.name1:
    layer_name = i
    render_deepdream(tf.square(T(layer_name)), img0)

for i in layer_dict.name2:
    layer_name = i
    render_deepdream(tf.square(T(layer_name)), img0)

for i in layer_dict.name3:
    layer_name = i
    render_deepdream(tf.square(T(layer_name)), img0)

print('all finish')
'''

