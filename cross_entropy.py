#!usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

def sigmoid(u):
    return 1/(1+np.exp(-u))

def softmax(u):
    e = np.exp(u)
    return e / np.sum(e)

def forward(x):
    global w1
    global w2
    u1 = x.dot(w1)
    z1 = sigmoid(u1)
    u2 = z1.dot(w2)
    y = softmax(u2)
    return y, z1

w1 = np.array([[0.1, 0.3], [0.2, 0.4]])
w2 = np.array([[0.1, 0.3], [0.2, 0.4]])

x = np.array([[1, 0.5]])
y, z1 = forward(x)

print z1
print y
