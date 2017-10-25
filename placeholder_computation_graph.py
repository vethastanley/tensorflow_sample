import tensorflow as tf
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_fun = a + b
sess = tf.Session()
print(sess.run(adder_fun, {a: 3, b: 4.5}))
print(sess.run(adder_fun, {a: [1, 3], b: [2, 4]}))
adder_mul_fun = adder_fun * 3
print(sess.run(adder_mul_fun, {a: 3, b: 4.5}))