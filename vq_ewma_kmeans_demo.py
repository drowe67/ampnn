#!/usr/bin/python3
'''
  Demo of a custom Vector Quantiser layer written in tf.keras.  It
  uses kmeans to train, with updates performed on each batch using
  moving averages.

  Refs:
  [1] VQ-VAE_Keras_MNIST_Example.ipynb
      https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb
  [2] "Neural Discrete Representation Learning", Aaron van den Oord etc al, 2018

'''

import logging
import os
import numpy as np
from matplotlib import pyplot as plt

# Give TF "a bit of shoosh" - nneds to be placed _before_ "import tensorflow as tf"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from vq_ewma_kmeans import *

# constants
dim = 2
nb_samples = 1000
nb_embedding = 4

# Simple test model

inputs = tf.keras.layers.Input(shape=(dim,))
outputs = VQ_EWMA_kmeans(dim,nb_embedding,name="vq")(inputs)

model = tf.keras.Model(inputs, outputs)
# note we do our own training (no trainable wieghts) so choices here don't matter much
model.compile(loss='mse',optimizer='adam')
model.summary()

# training data - a QPSK constellation with noise
bits = np.random.randint(2,size=nb_samples*dim).reshape(nb_samples, dim)
x_train = 2*bits-1 + 0.1*np.random.randn(nb_samples, dim)

# Set up initial VQ table to something we know should converge
vq_initial = np.array([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]])/10
model.get_layer('vq').set_vq(vq_initial)
print(model.get_layer('vq').get_vq().numpy())

model.fit(x_train, x_train, batch_size=2, epochs=2)
vq_entries = model.get_layer('vq').get_vq().numpy()
print(vq_entries)
plt.scatter(x_train[:,0],x_train[:,1])
plt.scatter(vq_entries[:,0],vq_entries[:,1], marker='x')
plt.show()

