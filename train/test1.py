import time
import tensorflow as tf
import numpy as np

data_set=tf.data.Dataset.from_tensor_slices(np.array([[i]*10000 for i in range(10010)],dtype=np.int64)).prefetch(1000)
data_set=data_set.map(lambda x:x*x-1).prefetch(1000)
iterater=data_set.make_one_shot_iterator()
next=iterater.get_next()

y=next+1
sess=tf.Session()

sess.run(y)
sess.run(y)
sess.run(y)
sess.run(y)
sess.run(y)

s=time.time()
for _ in range(10000):
    out=sess.run(y)
e=time.time()
print(e-s)