#!/usr/bin/python3
'''
  Demo of custom Vector Quantiser layer written in tf.keras:

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

# Section 3.2 of [2]: custom layer to copy gradient from decoder input z_q(x) to encoder output z_e(x)
# transparent layer (input = output), but stop any enc weights being changed based on VQ error,
# gradient feedback path for enc gradients over top of VQ
class add(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def call(self, inputs):
        z_q, z_e = inputs
        return z_e + tf.stop_gradient(z_q - z_e)

# Calculate vq-vae loss.
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = tf.losses.mse(x, x_hat) / data_variance
        
        e_latent_loss = tf.math.reduce_mean((tf.stop_gradient(quantized) - x_inputs) ** 2)
        
        return recon_loss + commitment_cost * e_latent_loss
    return vq_vae_loss

# constants
dim = 2
nb_samples = 1000
nb_embedding = 4
commitment_cost = 0.25

# training data - a QPSK constellation with noise
bits = np.random.randint(2,size=nb_samples*dim).reshape(nb_samples, dim)
x_train = 2*bits-1 + 0.1*np.random.randn(nb_samples, dim)

# VQ VAE model --------------------------------------------

'''
x = tf.keras.layers.Input(shape=(dim,))
# dummy layer would be an NN layer in practice
z_e = tf.keras.layers.Dense(dim,)(x)
#z_e = tf.keras.layers.Lambda(lambda x: x)(x)
z_q = VQ_kmeans(dim, nb_embedding, name="vq")(z_e)
# transparent layer (input = output), but stop any enc weights being changed based on VQ error,
# feedback path for enc gradients over top of VQ
z_q_ = tf.keras.layers.Lambda(lambda z_q: z_e + tf.stop_gradient(z_q - z_e))(z_q)

# Decoder
# just to show where decoder layer goes, will be trained to be identity
p_x = tf.keras.layers.Dense(dim, name='dec_dense')(z_q_)
'''

x = tf.keras.layers.Input(shape=(dim,))
# dummy layer would be an NN layer in practice
z_e = tf.keras.layers.Lambda(lambda x: x)(x)
#z_e = tf.keras.layers.Dense(dim)(x)
z_q = VQ_kmeans(dim, nb_embedding, name="vq")(z_e)
# transparent layer (input = output), but stop any enc weights being changed based on VQ error,
# feedback path for enc gradients over top of VQ
z_q_ = add()([z_q, z_e])
#z_q_ = tf.keras.layers.Lambda(lambda z_q: z_q)(z_q)
p_x = tf.keras.layers.Dense(dim, name='dec_dense')(z_q_)

model = tf.keras.Model(x, p_x)
data_variance = np.var(x_train)

# loss = vq_vae_loss_wrapper(data_variance, commitment_cost, z_q, z_e)
model.compile(loss='mse', optimizer='adam')
model.summary()

# Set up initial VQ table to something we know should converge
vq_initial = np.array([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]])/10
model.get_layer('vq').set_vq(vq_initial)
print(model.get_layer('vq').get_vq())

model.fit(x_train, x_train, batch_size=2, epochs=2)
vq_entries = model.get_layer('vq').get_vq()
print(vq_entries)
plt.scatter(x_train[:,0],x_train[:,1])
#plt.scatter(vq_entries[:,0],vq_entries[:,1], marker='x')
#plt.show()

