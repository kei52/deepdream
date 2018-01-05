# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

#!wget -nc https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip -n inception5h.zip

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

import tensorflow as tf
import numpy as np

a = tf.Variable(tf.constant([4.]), name = 'a' )

w = tf.Variable(tf.constant([3.]), name='w')
b = tf.Variable(tf.constant([1.]), name='b')
x = tf.Variable(tf.constant([2.]), name='x')
y_ = tf.Variable(tf.constant([5.]), name='y_')
image_r = tf.Variable(tf.constant([244.]), name='image_r')
#print w,b,x,y_
image_g = tf.Variable(tf.constant([233.]), name='image_g')
image_b = tf.Variable(tf.constant([222.]), name='image_b')
image_rgb = tf.Variable(tf.constant([244.,233.,222.]), name='image_rgb')
relu = tf.nn.relu(image_rgb)
softmax = tf.nn.softmax(image_rgb)
log = tf.log(image_rgb)
p = w*x #tf.Variable(tf.constant([3.]), name='w') * tf.Variable(tf.constant([2.]), name='x')
y = p+b
s = -y
t = s +y_
f = t*t
gradient_test = a*image_rgb
#f = -(tf.Variable(tf.constant([3.]), name='w') * tf.Variable(tf.constant([2.]), name='x')+tf.Variable(tf.constant([1.]), name='b')) + tf.Variable(tf.constant[5.],name='y')
#f = -tf.Variable(tf.constant([3.]), name='w') * tf.Variable(tf.constant([2.]), name='x') - tf.Variable(tf.constant([1.]), name='b') + tf.Variable(tf.constant[5.],name='y')
'''
print 'p:',p
print 'y:',y
print 's:',s
print 't:',t
print 'f:',f
'''
gx, gb, gw, gp, gy, gy_,gs, gt, gf = tf.gradients(f, [x, b, w, p, y, y_,s, t, f])
gradient_softmax = tf.gradients(softmax,image_rgb)
gradient_imagergb = tf.gradients(gradient_test,image_rgb)

'''
print 'x:',gx #x
print 'b:',gb #b
print 'w:',gw #w
print 'p:',gp #p
print 'y:',gy #y
print 'y_:',gy_ #y_
print 's:',gs
print 't:',gt
print 'f:',gf
'''
init = tf.global_variables_initializer()

opt = tf.train.GradientDescentOptimizer(1.0)
train = opt.minimize(f)

with tf.Session() as sess:
    sess.run(init)
    print 'log',sess.run(log)
    print 'relu',sess.run(relu)
    print 'softmax',sess.run(softmax)
    print 'gradient softmax',sess.run(gradient_softmax)
    print 'gradient constant',sess.run(gradient_imagergb)
    print '---------- initialvariables ----------'
    print 'x:%.2f, w:%.2f, b:%.2f' % (sess.run(x), sess.run(w), sess.run(b))
    print 'p:%.2f, y:%.2f, y_:%.2f'% (sess.run(p), sess.run(y), sess.run(y_))
    print 's:%.2f, t:%.2f, f:%.2f' % (sess.run(s), sess.run(t), sess.run(f))
    print 'image rgb',sess.run(image_rgb)
    print '\n'
    
    print '---------- gradient ----------'
    print 'gx:%.2f, gw:%.2f, gb: %.2f' % (sess.run(gx), sess.run(gw), sess.run(gb))
    print 'gp:%.2f, gy:%.2f, gy_:%.2f' %(sess.run(gp), sess.run(gy), sess.run(gy_))
    print 'gs:%.2f, gt:%.2f, gf:%.2f' %(sess.run(gs), sess.run(gt), sess.run(gf))
    print '\n'
    
    print '---------- run GradientDescentOptimizer ----------'
    sess.run(train)
    print 'x:%.2f, w:%.2f, b:%.2f' % (sess.run(x), sess.run(w), sess.run(b))
    print 'p:%.2f, y:%.2f, y_:%.2f'% (sess.run(p), sess.run(y), sess.run(y_))
    print 's:%.2f, t:%.2f, f:%.2f'%(sess.run(s), sess.run(t), sess.run(f))
    print '\n'

    print '---------- gradient ----------'
    print 'gx:%.2f, gw:%.2f, gb: %.2f' % (sess.run(gx), sess.run(gw), sess.run(gb))
    print 'gp:%.2f, gy:%.2f, gy_:%.2f' %(sess.run(gp), sess.run(gy), sess.run(gy_))
    print 'gs:%.2f, gt:%.2f, gf:%.2f' %(sess.run(gs), sess.run(gt), sess.run(gf))
    
