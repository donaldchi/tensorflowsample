# coding: utf-8
import tensorflow as tf

matrix1 = tf.constant([[3,3]]) # create matrix
matrix2 = tf.constant([[2], [2]])

product = tf.matmul(matrix1, matrix2)

# method 1
sess = tf.Session()
result1 = sess.run(product)
print(result1)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


