import tensorflow as tf

# Model parameters
a = tf.Variable([.1], dtype=tf.float32)
b = tf.Variable([-.1], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = a * x + b
l = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - l)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
l_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x: x_train, l: l_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([a, b, loss], {x: x_train, l: l_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))