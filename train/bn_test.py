import time
import tensorflow as tf
import numpy as np

data_set=tf.data.Dataset.from_tensor_slices(np.array([0,1,2,3,4,5,6]))
data_set=data_set.map(lambda x:x+10)
data_set=data_set.shuffle(buffer_size=2)
data_set=data_set.repeat()
iterater=data_set.make_one_shot_iterator()
next=iterater.get_next()

sess=tf.Session()
for _ in range(100):
    out=sess.run(next)