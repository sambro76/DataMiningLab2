from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
"""Input layer"""
x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.random_normal([784, 1000]))
b1 = tf.Variable(tf.zeros([1000]))
"Summation Z"
z1 = tf.matmul(x, W1) + b1

"""1st hidden layer"""
#h1 = tf.nn.relu(z1)
h1 = tf.nn.sigmoid(z1)
W2 = tf.Variable(tf.random_normal([1000, 400]))
b2 = tf.Variable(tf.zeros([400]))
z2 = tf.matmul(h1, W2) + b2

"""the below codes are not included. This is for demo only. For more information, please contact the owner"""
