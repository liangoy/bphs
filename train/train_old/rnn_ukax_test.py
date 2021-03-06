import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell

long = 30
batch_size = 512

data_ax = pd.read_csv('/usr/local/oybb/project/bphs/data/ax.csv').dropna()
data_uk = pd.read_csv('/usr/local/oybb/project/bphs/data/uk.csv').dropna()

data = pd.merge(data_uk, data_ax, on='Date', how='left').sort_values(by='Date')
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

data_train = data[: - long + -7]
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
        a.append(sample[:-1, :5])
        b.append(sample[:-1, 5:10])
        c.append(sample[-1][1])
    return a, b, c


x = tf.placeholder(shape=[batch_size, long - 1, 5], dtype=tf.float16)
y = tf.placeholder(shape=[batch_size, long - 1, 5], dtype=tf.float16)
z_ = tf.placeholder(shape=[batch_size], dtype=tf.float16)

X = tf.nn.sigmoid(x) - 0.5
Y = tf.nn.sigmoid(y) - 0.5

gru_x = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x = gru_x.zero_state(batch_size, dtype=tf.float16)
with tf.variable_scope('RNN_x'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x, state_x) = gru_x(X[:, timestep], state_x)
    out_put_x = state_x

gru_y = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_y = gru_y.zero_state(batch_size, dtype=tf.float16)
with tf.variable_scope('RNN_y'):
    for timestep in range(long - 1):  # be careful
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_y, state_y) = gru_y(Y[:, timestep], state_y)
    out_put_y = state_y

out_put = tf.concat([out_put_x, out_put_y], axis=1)

lay1 = ml.layer_basic(out_put, 4)
z = ml.layer_basic(lay1, 1)[:, 0]

loss = tf.reduce_mean((z - z_) ** 2)

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
        z_train, z_train_, loss_train = sess.run((z, z_, loss), feed_dict={x: a, y: b, z_: c})
        z_test, z_test_, loss_test = sess.run((z, z_, loss), feed_dict={x: a_test, y: b_test, z_: c_test})
        q_train = np.mean(np.abs(z_train - z_train_))
        q_test = np.mean(np.abs(z_test - z_test_))
        print(loss_train, loss_test, q_train, q_test,np.corrcoef(z_test,z_test_)[0,1])