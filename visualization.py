# coding: utf-8
# tensorboard --logdir='logs/'
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = 'layer{}'.format(n_layer)
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')  # row: in_size, column: out_size
            tf.summary.histogram(layer_name + '/weights', Weights)  # tensorflow >= 0.12
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1) # row: 1, column: out_size
            tf.summary.histogram(layer_name + '/biases', Weights)  # tensorflow >= 0.12
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', Weights)  # tensorflow >= 0.12
        return outputs

# create network structure
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# define layer in network
l1 = add_layer(xs, 1, 10, 1, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, 2, activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                         reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

# start training: input data
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()
sess.run(init)


for step in range(1000):
    # traning
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if step%50==0:
        rs = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(rs, step)