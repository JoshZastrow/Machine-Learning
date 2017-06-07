import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # shut off stupid warnings
sess = tf.Session()

# Create computational nodes -- constants
node_1 = tf.constant(3.0, tf.float32)
node_2 = tf.constant(4.0)
node_3 = tf.add(node_1, node_2)
node_4 = tf.multiply(node_3, node_1)

# Create paramaterized nodes (give inputs later)
a = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32)

node_add = a + c
node_add_triple = node_add * node_1

# Session Results
print('contant input session:'
      '\t(3.0 + 4.0) * 3.0'
      '\n\ta = 3.0 \n\tb = 4.0 \n',
      sess.run(node_4))

print('parameterized session:'
      '\ta + b'
      '\n\ta = 3, 1\n\tb = 4, 2\n',
      sess.run(node_add, {a: [3.0, 1.0], c: [4.0, 2.0]}))


# Create Variables
W = tf.Variable([2.0], tf.float32)
b = tf.Variable([-1.0], tf.float32)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

x_train = [1, 2, 3, 4]
y_train = [3, 7, 11, 15]

# Model and loss function
linear_model = W * x + b
square_delta = tf.square(linear_model - y)
loss = tf.reduce_sum(square_delta)

# create handle to Tensorflow subgraph that can initialze all variabls
init = tf.global_variables_initializer()
sess.run(init)

# Training
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print('\nresults of session:\n\tW: {}\n\tb: {}\n\tloss: {}'
      .format(curr_W, curr_b, curr_loss))

