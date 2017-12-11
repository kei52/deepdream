#!usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

#活性化関数一層目にはシグモイド関数を用いる
def sigmoid(u):
    return 1/(1+np.exp(-u))
#活性化関数二層目にはソフトマックス関数を用いる
def softmax(u):
    e = np.exp(u)
    return e / np.sum(e)
#ネット値uと実際に層の構成を行う
def forward(x):
    global W1
    global W2
    u1 = x.dot(W1)
    print 'u1:',u1
    z1 = sigmoid(u1)
    u2 = z1.dot(W2)
    y = softmax(u2)
    return y, z1
'''
x = np.array([[x1, x2]])
となります。

W1、W2はネットワークのパラメータの行列で、 

W1 = np.array([[w1_11, w1_21], [w1_12, w1_22]])
'''
#Wの初期値を決定する←この値を更新していく
W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])

#xの値（入力）の定義
x = np.array([[1, 0.5]])
#一層目の出力z1、二層目の出力yを求める
y, z1 = forward(x)

print 'z1:',z1
print 'y:',y

#誤差関数としてクロスエントロピー誤差を用いる
def back_propagation(x,z1,y,d):
    #更新するwを求める
    global W1
    global W2
    #誤差関数をdelta2とする
    #dとは？
    delta2 = y - d
    grad_W2 = z1.T.dot(delta2)
    
    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    grad_W1 = x.T.dot(delta1)
    #learning_rateは学習係数 grad
    W2 = W2 - learning_rate * grad_W2
    W1 = W1 - learning_rate * grad_W1

W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])
learning_rate = 0.005

# 順伝播
x = np.array([[1, 0.5]])
y, z1 = forward(x)

# 誤差逆伝播
d = np.array([[1, 0]]) # 教師データ
back_propagation(x, z1, y, d)

print 'W1:',W1
print 'W2:',W2


