import tensorflow as tf

# Training Data
x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

# Try to find values for W and b that compute y_data = W * x_data + b
# We know that W should be 1 and b 0
# random_uniform -> random function -1.0 ~ 1.0
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Place Holder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our Hypothesis H(x) = Wx+b
hypothesis = W * X + b

# Simplified cost function Cost(Loss) = 1/m * Sigma (H(xi)-yi)2
# tf.square = x^2
cost = tf.reduce_mean(tf.square(hypothesis-Y))

# Minimize -> Gradient Descent Algorithm
a = tf.Variable(0.1)    # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before Starting, initalize the Variables
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 30 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b)

# Leanrs best fit is W : [0.1], b : [0.3]
