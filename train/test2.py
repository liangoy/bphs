import time
import tensorflow as tf
import numpy as np

next = tf.placeholder(shape=[10000], dtype=tf.int64)

y=next+1
sess=tf.Session()

sess.run(y, feed_dict={next: [1]*10000})
sess.run(y, feed_dict={next: [1]*10000})
sess.run(y, feed_dict={next: [1]*10000})
sess.run(y, feed_dict={next: [1]*10000})
sess.run(y, feed_dict={next: [1]*10000})

lis=np.array([[i]*10000 for i in range(10000)],dtype=np.int64)
s=time.time()
for i in lis:
    out=sess.run(y, feed_dict={next: i*i-1})
e = time.time()
print(e - s)
