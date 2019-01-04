import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell

long = 7
batch_size = 256

data_bp = pd.read_csv('/home/liangoy/Desktop/project/bphs/data/bp.csv').dropna()
data_hs = pd.read_csv('/home/liangoy/Desktop/project/bphs/data/hs.csv').dropna()

data = pd.merge(data_bp, data_hs, on='Date', how='outer').sort_values(by='Date')
data = data.fillna(method='ffill')

for i in data.columns[1:]:
    std = data[i].std()
    mean = data[i].mean()
    data[i] = (data[i] - mean) / std

data = np.array(data)

data_train = data[:-1 * batch_size - long]
data_test = data[-1 * batch_size - long:]


# ['Date', 'Open_x', 'High_x', 'Low_x', 'Close_x', 'Adj Close_x','Volume_x', 'Open_y', 'High_y', 'Low_y', 'Close_y', 'Adj Close_y','Volume_y']
def next(bs=batch_size):
    r = np.random.randint(0, len(data_train) - long, bs)
    a, b, c = [], [], []
    for i in r:
        sample = data_train[i: i + long]
        sample = np.array(sample)
        a.append(sample[:-1, 1:7])
        b.append(sample[:, 7:])
        c.append(sample[-1][4])
    return a, b, c


###########################################################
def next_test(bs=batch_size):
    #r = np.random.randint(0, len(data_test) - long, bs)
    r = range(len(data_test) - long)
    a, b, c = [], [], []
    for i in r:
        sample = data_test[i: i + long]
        sample = np.array(sample)
        a.append(sample[:-1, 1:7])
        b.append(sample[:, 7:])
        c.append(sample[-1][4])
    return a, b, c


##########################################################

x = tf.placeholder(shape=[batch_size, long - 1, 6], dtype=tf.float16)
y = tf.placeholder(shape=[batch_size, long, 6], dtype=tf.float16)
z_ = tf.placeholder(shape=[batch_size], dtype=tf.float16)

gru = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x = gru.zero_state(batch_size, dtype=tf.float16)
with tf.variable_scope('RNN_x'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x, state_x) = gru(x[:, timestep], state_x)
    out_put_x = state_x

state_y = gru.zero_state(batch_size, dtype=tf.float16)
with tf.variable_scope('RNN_y'):
    for timestep in range(long):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_y, state_y) = gru(y[:, timestep], state_y)
    out_put_y = state_y

out_put = tf.concat([out_put_x, out_put_y], axis=1)

lay1 = tf.nn.elu(ml.layer_basic(out_put, 4))
z = ml.layer_basic(lay1, 1)[:, 0]+x[:,-1,4]

loss = tf.reduce_mean((z - z_) ** 2)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................')

for i in range(10 ** 10):
    a, b, c = next()
    sess.run(optimizer, feed_dict={x: a, y: b, z_: c})
    if i % 100 == 0:
        a_test, b_test, c_test = next_test()
        train_loss = sess.run(loss, feed_dict={x: a, y: b, z_: c}) ** 0.5
        test_loss = sess.run(loss, feed_dict={x: a_test, y: b_test, z_: c_test}) ** 0.5
        print(train_loss,test_loss)
