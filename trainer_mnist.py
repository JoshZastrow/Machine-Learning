
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
Recreated code from tensorflow's fully_connected_feed.py -- educational purposes only
contains the training functions to train the model in model_mnist.py
'''

"""Trains and Evaluates the MNIST network using a feed dictionary."""


import argparse
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist
