import numpy as np
import tensorflow as tf
import math
import time
import os
import sys
sess = tf.Session()
Y = np.array([[0.1, 0.2, 0.3, 0.4],
              [0.0, 0.8, 0.2, 0.0],
              [0.0, 0.4, 0.5, 0.1]
             ])
print Y
Y_ = np.array([[0.0, 0.0, 1.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0]
             ])
print Y_

def argmax_test():
    print sess.run(tf.argmax(Y,1))
    print sess.run(tf.argmax(Y_,1))

def equal_test():
    equal = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))
    print sess.run(equal)
    equal_cast = tf.cast(equal,tf.float32)
    print sess.run(equal_cast)
def main():
    argmax_test()
    equal_test()

if __name__ == '__main__':
    main()
    
