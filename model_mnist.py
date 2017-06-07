'''
Recreated code from tensorflow's mnist.py -- educational purposes only
contains the forward, loss and backprop functions for the model

This file is used in other trainer files, and not intended to be run on it's own
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist


# Establish some variables that describe the data and model
num_classes = 10
image_size = 28
num_pixels = image_size ** 2

# Basic model parameters as external flags?
FLAGS = None


def main(_):

    # Get Data
    dataset = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create input layer parameters
    X = tf.placeholder(tf.float32, [None, num_pixels])
    Y = tf.placeholder(tf.float32, [None, num_classes])

    W = tf.Variable(tf.zeros([num_pixels, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Linear Classifier -- Forward
    y = tf.matmul(x, W) + b

    # Define loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))

    # Trainer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
