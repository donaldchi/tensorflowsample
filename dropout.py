# coding : utf-8
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data, X_train: 1257x64, y_train: 1257x10, X_test: 540x10, y_test: 540x10
digits = load_digits()
X = digits.data
y = digits.target

y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


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

        # add dropout function
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', Weights)  # tensorflow >= 0.12
        tf.summary.histogram(layer_name + '/outputs', biases)  # tensorflow >= 0.12
        return outputs


# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)  # drop_prob = 1 - keep_prob
xs = tf.placeholder(tf.float32, [None, 64])  # 8x8
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, 'l2', activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)


sess = tf.Session()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
sess.run(tf.global_variables_initializer())

for step in range(501):
    # training
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.5})
    if step%50==0:
        train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})
        train_writer.add_summary(train_result, step)
        test_writer.add_summary(test_result, step)
