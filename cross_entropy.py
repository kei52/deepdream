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
    print '----------forward----------'
    global W1
    global W2
    u1 = x.dot(W1)
    print 'u1:',u1
    z1 = sigmoid(u1)
    u2 = z1.dot(W2)
    y = softmax(u2)
    return y, z1
    print '\n'

#Wの初期値を決定する←この値を更新していく
W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])

#xの値（入力）の定義
x = np.array([[1, 0.5]])
#一層目の出力z1、二層目の出力yを求める
y, z1 = forward(x)

print '-----print-----'
print 'z1:',z1
print 'y:',y

#誤差関数としてクロスエントロピー誤差を用いる
def back_propagation(x,z1,y,d):
    print '----------back propagation----------'
    #更新するwを求める
    global W1
    global W2
    #誤差関数をdelta2とする
    #dは教師データ
    delta2 = y - d
    print 'delta2:',delta2
    grad_W2 = z1.T.dot(delta2)
    print 'grad_W2:',grad_W2
    sigmoid_dash = z1 * (1 - z1)
    delta1 = delta2.dot(W2.T) * sigmoid_dash
    grad_W1 = x.T.dot(delta1)
    print 'grad_W1:',grad_W1
    #learning_rateは学習係数 grad
    W2 = W2 - learning_rate * grad_W2
    W1 = W1 - learning_rate * grad_W1
    print 'W2:',W2
    print 'W1:',W1
    print 'finish back propagation\n'

W1 = np.array([[0.1, 0.3], [0.2, 0.4]])
W2 = np.array([[0.1, 0.3], [0.2, 0.4]])
learning_rate = 0.005

# 順伝播
# 入力の値ｘ
x = np.array([[1, 0.5]])
y, z1 = forward(x)

# 誤差逆伝播
d = np.array([[1, 0]]) # 教師データ
back_propagation(x, z1, y, d)

#学習したパラメータの値
print 'W1:',W1
print 'W2:',W2


