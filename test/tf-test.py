import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
# y = x*weights + biases
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = weights * x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases))
