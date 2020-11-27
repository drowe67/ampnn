#!/usr/bin/python3
'''
  Demo of a VQ VAE written in tf.keras 2.3

  $ ./vq_vae_kmeans_demo.py
'''

import logging
import os
import numpy as np
from matplotlib import pyplot as plt

# Give TF "a bit of shoosh" - needs to be placed _before_ "import tensorflow as tf"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from vq_kmeans import *

# constants
dim = 2
nb_samples = 1000
nb_embedding = 4

# training data - data in 4 clusters with some noise (OK it's QPSK!)
bits = np.random.randint(2,size=nb_samples*dim).reshape(nb_samples, dim)
x_train = 2*bits-1 + 0.1*np.random.randn(nb_samples, dim)

# VQ VAE model --------------------------------------------

x = tf.keras.layers.Input(shape=(dim,))
# dummy layer would be an NN layer in practice
z_e = tf.keras.layers.Lambda(lambda x: x)(x)
#z_e = tf.keras.layers.Dense(dim, name="enc_dense")(x)
z_q = VQ_kmeans(dim, nb_embedding, name="vq")(z_e)
z_q_ = CopyGradient()([z_q, z_e])
# toy decoder layer
p = tf.keras.layers.Dense(dim, name='dec_dense')(z_q_)

model = tf.keras.Model(x, p)

model.add_loss(commitment_loss(z_e, z_q))
model.compile(loss='mse', optimizer='adam')
model.summary()

# Set up initial VQ table to something we know should converge
vq_initial = np.array([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]])/10
model.get_layer('vq').set_vq(vq_initial)
print(model.get_layer('vq').get_vq())

model.fit(x_train, x_train, batch_size=2, epochs=10)
vq_entries = model.get_layer('vq').get_vq()
print(model.get_layer('dec_dense').get_weights())
plt.scatter(x_train[:,0],x_train[:,1])
plt.scatter(vq_entries[:,0],vq_entries[:,1], marker='x')
plt.show()

