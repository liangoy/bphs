import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell

long = 7
batch_size = 256

data_bp = pd.read_csv('/home/liangoy/Desktop/project/bphs/data/bp.csv')
data_hs = pd.read_csv('/home/liangoy/Desktop/project/bphs/data/hs.csv')

data = pd.merge(data_bp, data_hs, on='Date', how='outer')
data = data.fillna(method='ffilll')

for i in data.columns[1:]:
    std = data[i].std()
    mean = data[i].mean()
    data[i] = (data[i] - mean) / std

data = np.array(data)

data_train = data[:-1 * batch_size-long]
data_test = data[-1 * batch_size-long:]


# ['Date', 'Open_x', 'High_x', 'Low_x', 'Close_x', 'Adj Close_x','Volume_x', 'Open_y', 'High_y', 'Low_y', 'Close_y', 'Adj Close_y','Volume_y']
def next(bs=batch_size):
    r = np.random.randint(0, len(data_train) - long, bs)
    a, b, c = [], [], []
    for i in r:
        sample = data_train[i: i + long]
        sample = np.array(sample)
        a.append(sample[:-1, 1:7])
        b.append(sample[:, 7:])
        c.append(1 if sample[-1][4] - sample[-2][4] > 0 else 0)
    return a, b, c


###########################################################
def next_test(bs=batch_size):
    # r = np.random.randint(0, len(data_test) - long, bs)
    r = range(len(data_test) - long)
    a, b, c = [], [], []
    for i in r:
        sample = data_test[i: i + long]
        sample = np.array(sample)
        a.append(sample[:-1, 1:7])
        b.append(sample[:, 7:])
        c.append(1 if sample[-1][4] - sample[-2][4] > 0 else 0)
    return a, b, c


##########################################################

x = tf.placeholder(shape=[batch_size, long - 1, 6], dtype=tf.float32)
y = tf.placeholder(shape=[batch_size, long, 6], dtype=tf.float32)
z_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

gru = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x = gru.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_x'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x, state_x) = gru(x[:, timestep], state_x)
    out_put_x = state_x

state_y = gru.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_y'):
    for timestep in range(long):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_y, state_y) = gru(y[:, timestep], state_y)
    out_put_y = state_y

out_put = tf.concat([out_put_x, out_put_y], axis=1)

lay1 = out_put_y+out_put_x

z = tf.nn.sigmoid(ml.layer_basic(lay1, 1)[:, 0])

loss = tf.reduce_sum(-z_ * tf.log(z + 0.000000001) - (1 - z_) * tf.log(1 - z + 0.00000001)) / batch_size / tf.log(2.0)
gv = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.1, scope=None), weights_list=gv)
all_loss = loss + l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('begin..................................')

for i in range(10 ** 10):
    a, b, c = next()
    sess.run(optimizer, feed_dict={x: a, y: b, z_: c})
    if i % 100 == 0:
        a_test, b_test, c_test = next_test()
        z_train,z__train,loss_train = sess.run((z,z_,loss), feed_dict={x: a, y: b, z_: c})
        z_test,z__test,loss_test = sess.run((z,z_,loss), feed_dict={x: a_test, y: b_test, z_: c_test})
        q_train=sum([1 for i in z_train+z__train if i<0.5 or i>1.5])/len(z_train)
        q_test = sum([1 for i in z_test + z__test if i < 0.5 or i > 1.5]) / len(z_test)
        print(loss_train,loss_test,q_train,q_test)
