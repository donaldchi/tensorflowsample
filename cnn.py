# coding: utf-8

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys, keep_prob: 1})
    return result

def weights_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv_2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # ksize = pooling size
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])/255.  # 28x28
ys = tf.placeholder(tf.float32, [None, 10])  # 28x28
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1]) # #channel=1, because the sample is gray scale graph
# print(x_image.shape)

## convolution layer 1 ##
W_conv1 = weights_variable([5,5,1,32])  # patch 5x5, in size 1 (input image depth), out size: 32 (output image depth)
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)  # the same as Wx + b, Wx part processed by tf.nn.conv2d
                                                           # output size: 28x28x32 because same padding be used
h_pool1 = max_pool_2x2(h_conv1)                            # output size : 14x14x32

## convolution layer 2 ##
W_conv2 = weights_variable([5,5,32,64])  # patch 5x5, in size 1 (input image depth), out size: 32 (output image depth)
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)  # the same as Wx + b, Wx part processed by tf.nn.conv2d
                                                           # output size: 14x14x64 because same padding be used
h_pool2 = max_pool_2x2(h_conv2)                            # output size : 7x7x64

## full connection layer 1 ##
W_f1 = weights_variable([7*7*64, 1024])
b_f1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # [n_samples, 7,7,64] => [n_samples, 7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_f1) + b_f1)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

## full connection layer 2 ##
W_f2 = weights_variable([1024, 10])
b_f2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_f2)+b_f2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys, keep_prob:0.5})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images[:1000], mnist.test.labels[:1000]))
