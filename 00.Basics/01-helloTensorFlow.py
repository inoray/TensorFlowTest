import tensorflow as tf

hello = tf.constant ("Hello, tensorflow.")
sess = tf.Session()

print(sess.run(hello))

a = tf.constant(2016)
b = tf.constant(10)

print(sess.run(a+b))
