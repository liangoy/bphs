import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell

long = 30
batch_size = 512

data_bp = pd.read_csv('/usr/local/oybb/project/bphs/data/bp.csv')
data_hs = pd.read_csv('/usr/local/oybb/project/bphs/data/hs.csv')

data = pd.merge(data_bp, data_hs, on='Date', how='outer').sort_values(by='Date')
data = data.fillna(method='ffill')

data = np.array(data)[:-1, 1:]
data = np.array(data, dtype=np.float32)
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

data_train = data[:-1 * batch_size - long + 1]
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
        a.append(sample[:-1, :11])
        b.append(sample[:-1, :11])
        c.append(sample[-1][:4])
    return a, b, c


x = tf.placeholder(shape=[batch_size, long - 1, 10], dtype=tf.float32)
y = tf.placeholder(shape=[batch_size, long - 1, 10], dtype=tf.float32)
z_ = tf.placeholder(shape=[batch_size, 4], dtype=tf.float32)

X = tf.nn.sigmoid(x) - 0.5
Y = tf.nn.sigmoid(y) - 0.5

gru_x_open = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x_open = gru_x_open.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_x_open'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x_open, state_x_open) = gru_x_open(X[:, timestep], state_x_open)
    out_put_x_open = state_x_open

gru_x_high = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x_high = gru_x_high.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_x_high'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x_high, state_x_high) = gru_x_high(X[:, timestep], state_x_high)
    out_put_x_high = state_x_high

gru_x_low = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x_low = gru_x_low.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_x_low'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x_low, state_x_low) = gru_x_low(X[:, timestep], state_x_low)
    out_put_x_low = state_x_low

gru_x_close = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x_close = gru_x_close.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_x_close'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x_close, state_x_close) = gru_x_close(X[:, timestep], state_x_close)
    out_put_x_close = state_x_close

z_open = ml.layer_basic(out_put_x_open, 1)[:, 0]
z_high = ml.layer_basic(out_put_x_high, 1)[:, 0]
z_low = ml.layer_basic(out_put_x_low, 1)[:, 0]
z_close = ml.layer_basic(tf.nn.elu(ml.layer_basic(out_put_x_close, 4)), 1)[:, 0]

z_open_, z_high_, z_low_, z_close_ = z_[:, 0], z_[:, 1], z_[:, 2], z_[:, 3]

loss_open = tf.reduce_mean((z_open - z_open_) ** 2)
loss_high = tf.reduce_mean((z_high - z_high_) ** 2)
loss_low = tf.reduce_mean((z_low - z_low_) ** 2)
loss_close = tf.reduce_mean((z_close - z_close_) ** 2)

# loss = (loss_open + loss_high + loss_low + loss_close) / 4
loss = loss_close
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
optimizer_min = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................')

for i in range(10 ** 10):
    a, b, c = next(data=data_train)
    sess.run(optimizer, feed_dict={x: a, y: b, z_: c})
    if i % 100 == 0:
        a_test, b_test, c_test = next(data=data_test, random=False)
        loss_train = sess.run(loss, feed_dict={x: a, y: b, z_: c})
        loss_test = sess.run(loss, feed_dict={x: a_test, y: b_test, z_: c_test})
        print(loss_train, loss_test)
