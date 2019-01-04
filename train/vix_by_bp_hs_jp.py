import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell
from sklearn.utils import shuffle
from config import ROOT_PATH

data_vix = pd.read_csv(ROOT_PATH + '/data/bp.csv').dropna()
data_vix = data_vix.drop(['Adj Close', 'Volume'], axis=1)
data_hs = pd.read_csv(ROOT_PATH + '/data/hs.csv').dropna()
data_hs = data_hs.drop(['Adj Close', 'Volume'], axis=1)
data_bp = pd.read_csv(ROOT_PATH + '/data/jp.csv').dropna()
data_bp = data_bp.drop(['Adj Close', 'Volume'], axis=1)
data_bp = pd.read_csv(ROOT_PATH + '/data/vix.csv').dropna()
data_bp = data_bp.drop(['Adj Close', 'Volume'], axis=1)

long = 20
batch_size = 1024
otype = 1

data = pd.merge(data_vix, data_hs, on='Date', how='left')
data = pd.merge(data, data_bp, on='Date', how='left')
data = pd.merge(data, data_bp, on='Date', how='left').sort_values(by='Date')
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
import pandas as pd
import numpy as np
import tensorflow as tf
from util import ml
from tensorflow.contrib.rnn import GRUCell
from sklearn.utils import shuffle
from config import ROOT_PATH

data_vix = pd.read_csv(ROOT_PATH + '/data/bp.csv').dropna()
data_vix = data_vix.drop(['Adj Close', 'Volume'], axis=1)
data_hs = pd.read_csv(ROOT_PATH + '/data/hs.csv').dropna()
data_hs = data_hs.drop(['Adj Close', 'Volume'], axis=1)
data_jp = pd.read_csv(ROOT_PATH + '/data/jp.csv').dropna()
data_jp = data_jp.drop(['Adj Close', 'Volume'], axis=1)
data_bp = pd.read_csv(ROOT_PATH + '/data/vix.csv').dropna()
data_bp = data_bp.drop(['Adj Close', 'Volume'], axis=1)

long = 20
batch_size = 1024
otype = 1

data = pd.merge(data_vix, data_hs, on='Date', how='left')
data = pd.merge(data, data_jp, on='Date', how='left')
data = pd.merge(data, data_bp, on='Date', how='left').sort_values(by='Date')
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


def next(data, bs=batch_size, random=True):
    if random:
        r = np.random.randint(0, len(data) - (long + 1), bs)
    else:
        r = range(bs)
    a, b = [], []
    for i in r:
        sample = data[i: i + long + 1]
        a.append(sample[:-1, :])
        b.append(sample[-1, otype])
    return a, b


x = tf.placeholder(shape=shape, dtype=tf.float16)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float16)

gru = GRUCell(num_units=16, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state = gru.zero_state(batch_size, dtype=tf.float16)
with tf.variable_scope('RNN'):
    for timestep in range(long):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = gru(x[:, timestep], state)
    out_put = state

y = ml.layer_basic(out_put, 1)[:, 0]

loss = tf.reduce_mean((y - y_) ** 2)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
optimizer_min = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    train_x, train_y = next(data=data_train)
    sess.run(optimizer, feed_dict={x: train_x, y_: train_y})
    if i % 100 == 0:
        test_x, test_y = next(data=data_test)
        loss_train = sess.run(loss, feed_dict={x: train_x, y_: train_y})
        y_test, y_test_, loss_test = sess.run((y, y_, loss), feed_dict={x: test_x, y_: test_y})
        print(loss_train, loss_test, np.mean(np.abs(y_test - y_test_)) / np.mean(np.abs(y_test_)),
              np.corrcoef(y_test, y_test_)[0, 1])

data_train = data[:-512]  # ???
data_test = data[-512:]


def next(data, bs=batch_size, random=True):
    if random:
        r = np.random.randint(0, len(data) - (long + 1), bs)
    else:
        r = range(bs)
    a, b = [], []
    for i in r:
        sample = data[i: i + long + 1]
        a.append(sample[:-1, :])
        b.append(sample[-1, otype])
    return a, b


x = tf.placeholder(shape=shape, dtype=tf.float16)
y_ = tf.placeholder(shape=[batch_size], dtype=tf.float16)

gru = GRUCell(num_units=16, reuse=tf.AUTO_REUSE, activation=tf.nn.elu)
state = gru.zero_state(batch_size, dtype=tf.float16)
with tf.variable_scope('RNN'):
    for timestep in range(long):
        if timestep == 1:
            tf.get_variable_scope().reuse_variables()
        (cell_output, state) = gru(ml.x[:, timestep], state)
    out_put = state

y = ml.layer_basic(out_put, 1)[:, 0]

loss = tf.reduce_mean((y - y_) ** 2)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
optimizer_min = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

# ...................................................................
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10 ** 10):
    train_x, train_y = next(data=data_train)
    sess.run(optimizer, feed_dict={x: train_x, y_: train_y})
    if i % 100 == 0:
        test_x, test_y = next(data=data_test)
        loss_train = sess.run(loss, feed_dict={x: train_x, y_: train_y})
        y_test, y_test_, loss_test = sess.run((y, y_, loss), feed_dict={x: test_x, y_: test_y})
        print(loss_train, loss_test, np.mean(np.abs(y_test - y_test_)) * 100,
              np.mean(np.abs(y_test - y_test_)) / np.mean(np.abs(y_test_)),
              np.corrcoef(y_test, y_test_)[0, 1])
