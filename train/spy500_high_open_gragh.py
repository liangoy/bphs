import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell
import tensorflow.contrib.tensorrt as trt
from config import ROOT_PATH

data_bp = pd.read_csv(ROOT_PATH + '/data/bp.csv').dropna()
data_bp = data_bp.drop(['Adj Close', 'Volume'], axis=1)
data_hs = pd.read_csv(ROOT_PATH + '/data/hs.csv').dropna()
data_hs = data_hs.drop(['Adj Close', 'Volume'], axis=1)
data_jp = pd.read_csv(ROOT_PATH + '/data/jp.csv').dropna()
data_jp = data_jp.drop(['Adj Close', 'Volume'], axis=1)
data_vix = pd.read_csv(ROOT_PATH + '/data/vix.csv').dropna()
data_vix = data_vix.drop(['Adj Close', 'Volume'], axis=1)

long = 20
batch_size = 1024
otype = 1

data = pd.merge(data_bp, data_hs, on='Date', how='left')
data = pd.merge(data, data_jp, on='Date', how='left')
data = pd.merge(data, data_vix, on='Date', how='left').sort_values(by='Date')
data = data.drop('Date', axis=1)
data = data.iloc[300:].replace(0, None)
data = data.fillna(method='ffill')

data = np.array(data)[1:]
data = np.array(data, dtype=np.float16)
data_t = data[1:]
data_t_1 = data[:-1] + 0.0000001

'''['Open_x', 'High_x', 'Low_x', 'Close_x', 'Open_y', 'High_y', 'Low_y','Close_y', 'Open_x', 'High_x', 'Low_x', 'Close_x', 'Open_y', 'High_y','Low_y', 'Close_y']'''
for i in range(15):
    data_t[:, i] /= data_t_1[:, i // 4 * 4 + 3]
data = data_t - 1

shape = [batch_size, long, len(data[0])]
data_train = data[:-512]  # ???
data_test = data[-512:]


def next(data, bs=batch_size, test=False):
    if not test:
        r = np.random.randint(0, len(data) - (long + 1), bs)
    else:
        data = np.array(data).tolist()
        data.append(data[-1])
        data = np.array(data)
        r = range(len(data) - (long + 1) - bs, len(data) - (long + 1))
    a1, a2, b = [], [], []
    for i in r:
        sample = data[i: i + long + 1]
        a1.append(sample[:-1, :])
        a2.append(sample[-1, 0])
        b.append(sample[-1, 1] - sample[-1, 0])
    return a1, a2, b


g = tf.Graph()
with g.as_default():
    x1 = tf.placeholder(shape=shape, dtype=tf.float16)
    x2 = tf.placeholder(shape=[batch_size], dtype=tf.float16)
    y_ = tf.placeholder(shape=[batch_size], dtype=tf.float16)
    training = tf.placeholder(dtype=tf.bool)

    X = tf.layers.batch_normalization(x1, training=True, scale=False, center=False, axis=[0, -1])
    # X=x1
    gru = GRUCell(num_units=4, reuse=tf.AUTO_REUSE, activation=tf.nn.elu,
                  kernel_initializer=tf.glorot_normal_initializer(),dtype=tf.float16)
    state = gru.zero_state(batch_size, dtype=tf.float16)
    with tf.variable_scope('RNN'):
        for timestep in range(long):
            if timestep == 1:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = gru(X[:, timestep], state)
        out_put = state

    out = tf.nn.relu(out_put)

    y = ml.layer_basic(out, 1)[:, 0]

    loss = tf.reduce_mean((y - y_) ** 2,name='loss')

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)
    init = tf.global_variables_initializer()
#
trt_graph = trt.create_inference_graph(g.as_graph_def(),['loss'],
                                       max_batch_size=batch_size,
                                       max_workspace_size_bytes=2 ** 20,
                                       precision_mode="FP16")

with g.as_default():
    tf.import_graph_def(trt_graph,name='')
# with g.as_default():
#     out = tf.import_graph_def(
#         graph_def=trt_graph,
#         return_elements=["loss"]
#     )
# ...................................................................

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.40)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),graph=g)
sess.run(init)

import time
s=time.time()
for i in range(10 ** 3):
    train_x1, train_x2, train_y = next(data=data_train)
    sess.run(loss, feed_dict={x1: train_x1, x2: train_x2, y_: train_y, training: True})
e=time.time()
print(e-s)
