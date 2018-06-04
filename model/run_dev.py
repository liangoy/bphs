import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell

long = 30
batch_size = 256

data_bp = pd.read_csv('/usr/local/oybb/project/bphs/data/bp.csv')
data_hs = pd.read_csv('/usr/local/oybb/project/bphs/data/hs.csv')

data = pd.merge(data_bp, data_hs, on='Date', how='outer')
data = data.dropna()

data=np.array(data)[:-1,1:]
data=data[1:]/(data[:-1]+0.0000001)-1

data_train = data[:-1 * batch_size - long]
data_test = data[-1 * batch_size - long:]


# ['Open_x', 'High_x', 'Low_x', 'Close_x', 'Adj Close_x','Volume_x', 'Open_y', 'High_y', 'Low_y', 'Close_y', 'Adj Close_y','Volume_y']
def next(bs=batch_size):
    r = np.random.randint(0, len(data_train) - long, bs)
    a, b, c = [], [], []
    for i in r:
        sample = data_train[i: i + long]
        sample = np.array(sample)
        a.append(sample[:-1, :6])
        b.append(sample[:, 6:])
        c.append(sample[-1][3])
    return a, b, c


###########################################################
def next_test(bs=batch_size):
    #r = np.random.randint(0, len(data_test) - long, bs)
    r = range(len(data_test) - long)
    a, b, c = [], [], []
    for i in r:
        sample = data_test[i: i + long]
        sample = np.array(sample)
        a.append(sample[:-1, :6])
        b.append(sample[:, 6:])
        c.append(sample[-1][3])
    return a, b, c


##########################################################

x = tf.placeholder(shape=[batch_size, long - 1, 6], dtype=tf.float32)
y = tf.placeholder(shape=[batch_size, long, 6], dtype=tf.float32)
z_ = tf.placeholder(shape=[batch_size], dtype=tf.float32)

X=tf.nn.sigmoid(x)-0.5
Y=tf.nn.sigmoid(y)-0.5

gru_x = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_x = gru_x.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_x'):
    for timestep in range(long - 1):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_x, state_x) = gru_x(X[:, timestep], state_x)
    out_put_x = state_x

gru_y = GRUCell(num_units=8, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state_y = gru_y.zero_state(batch_size, dtype=tf.float32)
with tf.variable_scope('RNN_y'):
    for timestep in range(long):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output_y, state_y) = gru_y(Y[:, timestep], state_y)
    out_put_y = state_y

out_put = tf.concat([out_put_x, out_put_y], axis=1)
#out_put=out_put_y#+out_put_y

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
    a, b, c = next()
    sess.run(optimizer, feed_dict={x: a, y: b, z_: c})
    if i % 100 == 0:
        a_test, b_test, c_test = next_test()
        z_train,z__train,loss_train = sess.run((z,z_,loss), feed_dict={x: a, y: b, z_: c})
        z_test,z__test,loss_test = sess.run((z,z_,loss), feed_dict={x: a_test, y: b_test, z_: c_test})
        q_train=np.mean(np.abs(z_train-z__train))
        q_test = np.mean(np.abs(z_test - z__test))
        print(loss_train,loss_test,q_train,q_test)
