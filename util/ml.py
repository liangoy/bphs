import tensorflow as tf
import pymongo
import jieba
from scipy.optimize import leastsq
import numpy as np


def bn(x):
    mean, var = tf.nn.moments(x, axes=[0])
    var += 0.1 ** 7
    hat = (x - mean) / tf.sqrt(var)
    return hat


def bn_with_wb(x):
    w = tf.Variable(tf.random_normal([x.shape[1].value]))
    b = tf.Variable(tf.random_normal([x.shape[1].value]))
    return bn(x) * w + b


def layer_basic(x, size=0, with_b=True):
    if not size:
        size = x.shape[1].value
    w = tf.Variable(tf.random_normal([x.shape[1].value, size],dtype=tf.float16))
    if with_b:
        b = tf.Variable(tf.random_normal([size],dtype=tf.float16))
        return tf.matmul(x, w) + b
    else:
        return tf.matmul(x, w)


def res(x):
    lay1 = tf.nn.elu(layer_basic(bn_with_wb(x)))
    lay2 = tf.nn.elu(layer_basic(bn_with_wb(lay1)))
    lay3 = tf.nn.elu(layer_basic(bn_with_wb(lay2)))
    lay4 = tf.nn.elu(layer_basic(bn_with_wb(lay3)))
    lay5 = tf.nn.elu(layer_basic(bn_with_wb(lay4)))
    return lay5 + x


def conv2d(input, conv_filter, stride=[1, 1, 1, 1], padding='SAME', ksize=None, pool_stride=[1, 1, 1, 1],
           pool_padding='SAME',nn=tf.nn.elu):
    w = tf.Variable(tf.random_uniform(conv_filter, -1.0, 1.0))
    b = tf.Variable(tf.random_uniform([conv_filter[-1]], -1.0, 1.0))
    conv2d_out = nn(tf.nn.conv2d(input, w, strides=stride, padding=padding) + b)
    return tf.nn.max_pool(conv2d_out, ksize=ksize, strides=pool_stride, padding=pool_padding)

def res_modify(x,y,x_to_modify):
    func=lambda p,x:p[0]*x+p[1]
    error=lambda p,x,y:func(p,x)-y
    par=leastsq(error,(0,0),args=(x,y))[0]
    x_to_modify=np.array(x_to_modify)
    print(par)
    return x_to_modify*par[0]+par[1]


# oo = list(pymongo.MongoClient().xingqiao.dataWithMsg.find())
# w2i = {i['word']: i['index'] for i in pymongo.MongoClient().xingqiao.w2iGt5000.find()}


# def test(text):
#     jc = list(jieba.cut(text))
#     l = ([w2i.get(i, 1) for i in jc] + [0] * 100)[:100]
#     l = [l] * 8192
#     return sess.run(y, feed_dict={x: l})[0]


