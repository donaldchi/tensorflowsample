# coding : utf-8
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

# init = tf.global_variables_initializer()
# the above statement not be needed while using placeholder
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.], input2:[2.]})) # placeholderに値を与えるため, feed_dict
