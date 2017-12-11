#!usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

def sigmoid(u):
    return 1/(1+np.exp(-u))

def softmax(u):
    e = np.exp(u)
    return e / np.sum(e)

def forward(x):
    global W1
    global W2
    u1 = x.dot(W1)
    print 'u1:',u1,u1.shape
    z1 = sigmoid(u1)
    print 'z1:',z1,z1.shape
    u2 = z1.dot(W2)
    y = softmax(u2)
    return y, z1
'''
x = np.array([[x1, x2]])
となります。

W1、W2はネットワークのパラメータの行列で、

W1 = np.array([[w1_11, w1_21], [w1_12, w1_22]])
'''
W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])

x = np.array([[1, 0.5]])
y, z1 = forward(x)

print z1
print y
