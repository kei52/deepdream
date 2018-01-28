#!usr/bin/env python
# -*- coding: utf-8 -*-
'''
端末から以下のコマンドを実行する。
1. free (現状確認)
2. sudo su
3. echo 3 > /proc/sys/vm/drop_caches
4. free (効果確認)
'''
# 使用するライブラリをインポート
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from sklearn.datasets import fetch_mldata
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# define network
# placeholder（input image x.shape，output y）
x_ = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y_ = tf.placeholder(tf.float32, shape=(None, 10))
# conv1
conv1_features = 20 # conv1 dimention20
max_pool_size1 = 2 # conv1
conv1_w = tf.Variable(tf.truncated_normal([5, 5, 1, conv1_features], stddev=0.1), dtype=tf.float32) # 畳み込み層1の重み←更新していく
conv1_b = tf.Variable(tf.constant(0.1, shape=[conv1_features]), dtype=tf.float32) # 畳み込み層1のバイアス
conv1_c2 = tf.nn.conv2d(x_, conv1_w, strides=[1, 1, 1, 1], padding="SAME") # 畳み込み層1-畳み込み
conv1_relu = tf.nn.relu(conv1_c2+conv1_b) # 畳み込み層1-ReLU
conv1_mp = tf.nn.max_pool(conv1_relu, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding="SAME") # 畳み込み層1-マックスプーリング
# 畳み込み層2
conv2_features = 50 # 畳み込み層2の出力次元数
max_pool_size2 = 2 # 畳み込み層2のマックスプーリングのサイズ
conv2_w = tf.Variable(tf.truncated_normal([5, 5, conv1_features, conv2_features], stddev=0.1), dtype=tf.float32) # 畳み込み層2の重み
conv2_b = tf.Variable(tf.constant(0.1, shape=[conv2_features]), dtype=tf.float32) # 畳み込み層2のバイアス
conv2_c2 = tf.nn.conv2d(conv1_mp, conv2_w, strides=[1, 1, 1, 1], padding="SAME") # 畳み込み層2-畳み込み
conv2_relu = tf.nn.relu(conv2_c2+conv2_b) # 畳み込み層2-ReLU
conv2_mp = tf.nn.max_pool(conv2_relu, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding="SAME") # 畳み込み層2-マックスプーリング
# 全結合層1
print x_.get_shape()[1]#28[2]28
result_w = x_.get_shape()[1] // (max_pool_size1*max_pool_size2)
result_h = x_.get_shape()[2] // (max_pool_size1*max_pool_size2)
fc_input_size = result_w * result_h * conv2_features # 畳み込んだ結果、全結合層に入力する次元数
fc_features = 500 # 全結合層の出力次元数（隠れ層の次元数）
s = conv2_mp.get_shape().as_list() # [None, result_w, result_h, conv2_features]
conv_result = tf.reshape(conv2_mp, [-1, s[1]*s[2]*s[3]]) # 畳み込みの結果を1*N層に変換
fc1_w = tf.Variable(tf.truncated_normal([fc_input_size.value, fc_features], stddev=0.1), dtype=tf.float32) # 重み
fc1_b = tf.Variable(tf.constant(0.1, shape=[fc_features]), dtype=tf.float32) # バイアス
fc1 = tf.nn.relu(tf.matmul(conv_result, fc1_w)+fc1_b) # 全結合層1
# 全結合層2
fc2_w = tf.Variable(tf.truncated_normal([fc_features, fc_features], stddev=0.1), dtype=tf.float32) # 重み
fc2_b = tf.Variable(tf.constant(0.1, shape=[fc_features]), dtype=tf.float32) # バイアス
fc2 = tf.nn.relu(tf.matmul(fc1, fc2_w)+fc2_b) # 全結合層2
# 全結合層3
fc3_w = tf.Variable(tf.truncated_normal([fc_features, 10], stddev=0.1), dtype=tf.float32) # 重み
fc3_b = tf.Variable(tf.constant(0.1, shape=[10]), dtype=tf.float32) # バイアス
#最後の最後に求まる出力
y = tf.matmul(fc2, fc3_w)+fc3_b
# クロスエントロピー誤差とは誤差関数や！期待値（実際の値（情報量）と確率）
#E = tf.reduce_mean((function(x,e,a,b,c,d)-y)**2,0)
#labels _yが確率でyが情報量？＝期待値
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# 勾配法　最急降下法で学習させている
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 正解率の計算
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 学習
EPOCH_NUM = 1    #default 5
BATCH_SIZE = 1000    #default 1000
# 教師データ
mnist = fetch_mldata('MNIST original', data_home='.')
mnist.data = mnist.data.astype(np.float32) # image data 784*70000 [[0-255, 0-255, ...], [0-255, 0-255, ...], ... ]
mnist.data /= 255 # 0-1に正規化する
mnist.target = mnist.target.astype(np.int32) # ラベルデータ70000
print 'ready mnist data'
# 教師データを変換
N = 60000
train_x, test_x = np.split(mnist.data,   [N]) # 教師データ
train_y, test_y = np.split(mnist.target, [N]) # テスト用のデータ
train_x = train_x.reshape((len(train_x), 28, 28, 1)) # (N, height, width, channel)
test_x = test_x.reshape((len(test_x), 28, 28, 1))
# ラベルはone-hotベクトルに変換する
train_y = np.eye(np.max(train_y)+1)[train_y]
test_y = np.eye(np.max(test_y)+1)[test_y]
saver = tf.train.Saver()
# 学習
print("Train")
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    with tf.name_scope('summary'):
        writer = tf.summary.FileWriter('/tmp/tensorboard-sample', sess.graph)
        tf.summary.scalar('loss', cross_entropy)
        merged = tf.summary.merge_all()

    
    for epoch in range(EPOCH_NUM):
        perm = np.random.permutation(N)
        total_loss = 0
        for i in range(0, N, BATCH_SIZE):
            batch_x = train_x[perm[i:i+BATCH_SIZE]]
            batch_y = train_y[perm[i:i+BATCH_SIZE]]
            cross_entropy.eval(feed_dict={x_: batch_x, y_: batch_y})
            total_loss = total_loss + cross_entropy
            train_step.run(feed_dict={x_: batch_x, y_: batch_y})
            print 'i:{},total loss:{},\n'.format(i,total_loss)
        test_accuracy = accuracy.eval(feed_dict={x_: test_x, y_: test_y})
        if (epoch+1) % 1 == 0:
            print 'epoch:\t{}\ttotal loss:\t{}\tvaridation accuracy:\t{}'.format(epoch+1, total_loss, test_accuracy)
    saver.save(sess,'/home/roboworks/Desktop/model.ckpt')
    # 予測
    print '\nPredict'
    idx = np.random.choice(70000-N, 10)
    for i in idx:
        # y = tf.matmul(fc2, fc3_w)+fc3_b
        pre = np.argmax(y.eval(feed_dict={x_: [test_x[i]]}))
        plt.figure(figsize=(1,1))
        plt.imshow(test_x[i].reshape(28,28), cmap=cm.gray_r)
        plt.show()
        print 'score:', pre, '\n'
        #print '=&gt;', pre, '\n'
