# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

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

input_image = 'asagao'

model_fn = 'tensorflow_inception_graph.pb'
graph = tf.Graph()

f = tf.gfile.FastGFile(model_fn, 'rb')s
