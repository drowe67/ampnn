#!/usr/bin/python3
"""
  From: VQ-VAE_Keras_MNIST_Example.ipynb
  https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

  Simple VQ-VAE Keras example
"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Dense, Lambda
from keras import losses
from keras import backend as K
from keras.utils import to_categorical
import os

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# VQ layer.
class VQVAELayer(Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                  shape=(self.embedding_dim, self.num_embeddings),
                                  initializer=self.initializer,
                                  trainable=True)
        
        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))
        
        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        # Metrics.
        #avg_probs = K.mean(encodings, axis=0)
        #perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))
        
        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

# Calculate vq-vae loss.
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat) / data_variance
        
        e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
        q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
        loss = q_latent_loss + commitment_cost * e_latent_loss
        
        return recon_loss + loss #* beta
    return vq_vae_loss

# Hyper Parameters.
epochs = 20
batch_size = 64
validation_split = 0.1

# VQ-VAE Hyper Parameters.
embedding_dim =  2     # dimension of embedding vectors
num_embeddings = 4     # VQ size
commitment_cost = 0.25 # Controls the weighting of the loss terms.

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None)

# some samples with noise
nb_samples = 10000;
bits = np.random.randint(2,size=nb_samples*embedding_dim).reshape(nb_samples, embedding_dim)
x_train = 2*bits-1 + 0.1*np.random.randn(nb_samples, embedding_dim)

# Encoder
input_shape = (embedding_dim, )
inputs = Input(shape=input_shape, name='encoder_input')
# dummy encoder layer, this would normally be dense/conv
enc = Lambda(lambda inputs: inputs)(inputs)
enc_inputs = enc
enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc)
x = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)

# Decoder
# just to show where decoder layer goes, will be trained to be indentity
x = Dense(2, name='dec_dense')(x)

# Autoencoder.
vqvae = Model(inputs, x)
data_variance = np.var(x_train)
loss = vq_vae_loss_wrapper(data_variance, commitment_cost, enc, enc_inputs)
vqvae.compile(loss=loss, optimizer='adam')
vqvae.summary()

# seed VQ entries otherwise random start leads to converging on poor VQ extries
vq = np.array([[1.0,1.0,-1.0,-1.0],[1.0,-1.0,1.0,-1.0]])/10
vqvae.get_layer('vqvae').set_weights([vq])

history = vqvae.fit(x_train, x_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_split=validation_split,
                    callbacks=[esc])
vq_entries = vqvae.get_layer('vqvae').get_weights()[0]
plt.scatter(x_train[:,0],x_train[:,1])
plt.scatter(vq_entries[0,:],vq_entries[1,:], marker='x')
print(vqvae.get_layer('dec_dense').get_weights()[0])
plt.show()

