import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]= "-1"    # Disable GPU
#sys.path.append('baselines')
sys.path.append('traci_env')
import traci_env
import tensorflow as tf
#from baselines.common.models import register

OBS_SPACE = (20, 5, 1)
ACT_SPACE = 11

BATCH_SIZE = 32
DISCOUNT = 0.9
UPDATE_RATE = 500
EXPLORE = 5
ALPHA = 0.6
EPS = 0.000001

demands = ['Low.txt', 'Med.txt', 'High.txt']
rush = [33900, 34200, 34200]
dead = [0, 0, 0]

""""@register("street_cnn")
def street_cnn(**conv_kwargs):
    def network_fn(X):
        activ = tf.nn.leaky_relu
        cv1 = tf.layers.conv2d(X, 64, 2)
        flat = tf.layers.flatten(cv1)
        fc1 = activ(tf.layers.dense(flat, 64))
        fc2 = activ(tf.layers.dense(fc1, 64))
        return fc2
    return network_fn"""
