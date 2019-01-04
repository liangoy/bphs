import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell

long = 30 + 1
batch_size = 512
otype = 1

data_bp = pd.read_csv('/usr/local/oybb/project/bphs/data/bp.csv').dropna()
data_jp = pd.read_csv('/usr/local/oybb/project/bphs/data/jp.csv').dropna()

data = pd.merge(data_jp, data_bp, on='Date', how='left').sort_values(by='Date')
data = data.fillna(method='ffill')

data = np.array(data)[1:, 1:]
data = np.array(data, dtype=np.float16)
data_t = data[1:]
data_t_1 = data[:-1] + 0.0000001

dopen = data_t[:, 0] / data_t_1[:, 3]
dhigh = data_t[:, 1] / data_t_1[:, 3]
dlow = data_t[:, 2] / data_t_1[:, 3]
dclose = data_t[:, 3] / data_t_1[:, 3]
dvolume = data_t[:, 5] / data_t_1[:, 5]
dopen_hs = data_t[:, 6] / data_t_1[:, 9]
dhigh_hs = data_t[:, 7] / data_t_1[:, 9]
dlow_hs = data_t[:, 8] / data_t_1[:, 9]
dclose_hs = data_t[:, 9] / data_t_1[:, 9]
dvolume_hs = data_t[:, 11] / data_t_1[:, 11]

data = np.concatenate([dopen, dhigh, dlow, dclose, dvolume, dopen_hs, dhigh_hs, dlow_hs, dclose_hs, dvolume_hs],
                      axis=0) - 1
data = np.reshape(data, [-1, 10], order='F')

# data_train = data[:-1 * batch_size - long + 1]
data_train = data[:-1 * long - 7]
data_test = data[-1 * batch_size - long + 1:]


# ['Open_x', 'High_x', 'Low_x', 'Close_x', 'Adj Close_x','Volume_x', 'Open_y', 'High_y', 'Low_y', 'Close_y', 'Adj Close_y','Volume_y']
def next(data, bs=batch_size, random=True):
    if random:
        r = np.random.randint(0, len(data) - long, bs)
    else:
        r = range(bs)
    a, b, c = [], [], []
    for i in r:
        sample = data[i: i + long]
        a.append(np.concatenate([sample[:-1, :10], [[sample[-1][0]]] * (long - 1)], axis=-1))
        b.append(sample[-1][otype])
    return a, b


x = tf.placeholder(shape=[batch_size, long - 1, 11], dtype=tf.float16)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float16)

X = tf.reshape(tf.nn.tanh(x), [batch_size, long - 1, x.shape[-1], 1])

c1 = ml.conv2d(X, conv_filter=[1, x.shape[-1], 1, 4], padding='VALID', ksize=[1, 1, 1, 1], pool_padding='VALID', nn=tf.nn.tanh)
c2 = ml.conv2d(c1, conv_filter=[4, 1, 4, 6], padding='SAME', ksize=[1, 6, 1, 1], pool_stride=[1, 5, 1, 1],
               pool_padding='SAME', nn=tf.nn.tanh)
c3 = ml.conv2d(c2, conv_filter=[3, 1, 6, 8], padding='SAME', ksize=[1, 6, 1, 1], pool_stride=[1, 6, 1, 1],
               pool_padding='VALID', nn=tf.nn.tanh)

out_put = tf.reshape(c3, shape=[batch_size, 8])

y = ml.layer_basic(out_put, 1)[:, 0]

loss = tf.reduce_mean((y - y_) ** 2)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
optimizer_min = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................')

for i in range(10 ** 10):
    a, b = next(data=data_train)
    sess.run(optimizer, feed_dict={x: a, y_: b})
    if i % 100 == 0:
        a_test, b_test = next(data=data_test, random=False)
        y_train, y_train_, loss_train = sess.run((y, y_, loss), feed_dict={x: a, y_: b})
        y_test, y_test_, loss_test = sess.run((y, y_, loss), feed_dict={x: a_test, y_: b_test})
        q_train = np.mean(np.abs(y_train - y_train_))
        q_test = np.mean(np.abs(y_test - y_test_))
        print(loss_train, loss_test, q_train, q_test, np.corrcoef(y_test, y_test_)[0][1])
