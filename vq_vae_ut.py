#!/usr/bin/python3
"""
  Unit test of VQ-VAE Keras layer, used to explore operation and support development

    $ ./vq_vae_ut.py

  [1] From: VQ-VAE_Keras_MNIST_Example.ipynb
      https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt
import argparse

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Dense, Lambda
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import LambdaCallback
import os

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_tensor(label, tensor):
    return tf.Print(tensor, [tensor], label, first_n=-1, summarize=1024)
 
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
        flat_inputs = print_tensor("flat_inputs = ", flat_inputs)

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        distances = print_tensor("distances= ", distances)
        
        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        encoding_indices = K.print_tensor(encoding_indices, "encoding_indices: ")
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

# Callback to plot VQ entries as they evolve
def cb():
    vq_entries = vqvae.get_layer('vqvae').get_weights()[0]
    plt.clf()
    plt.scatter(vq_entries[0,:],vq_entries[1,:], marker='x')
    plt.xlim([-3,3]); plt.ylim([-3,3])
    plt.draw()
    plt.pause(0.0001)
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: cb() )

# Hyper Parameters.
batch_size = 2
commitment_cost = 0.25 # Controls the weighting of the loss terms.

parser = argparse.ArgumentParser(description='VQ training test')
parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=2, help='Number of samples to train on')
parser.add_argument('--embedding_dim', type=int, default=2,  help='dimension of embedding vectors')
parser.add_argument('--num_embedding', type=int, default=2,  help='number of embedded vectors')
args = parser.parse_args()
dim = args.embedding_dim
nb_samples = args.nb_samples;

x_train = b = np.asarray([[1,1],
                          [-1,-1]])
    
# Encoder
input_shape = (dim, )
inputs = Input(shape=input_shape, name='encoder_input')
# dummy encoder layer, this would normally be dense/conv
enc = Lambda(lambda inputs: inputs)(inputs)
enc_inputs = enc
enc = VQVAELayer(dim, args.num_embedding, commitment_cost, name="vqvae")(enc)
# transparent layer (input = output), but stop any weights being changed based on VQ error
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

# seed VQ entries with perfect vectors
vq = np.transpose(x_train)
vqvae.get_layer('vqvae').set_weights([vq])

print(x_train)
print(np.transpose(vq))

history = vqvae.fit(x_train, x_train,
                    batch_size=batch_size, epochs=args.epochs,
                    callbacks=[print_weights])
vq_entries = vqvae.get_layer('vqvae').get_weights()[0]
print(np.transpose(vq_entries))

#plt.scatter(x_train[:,0],x_train[:,1])
#plt.scatter(vq_entries[0,:],vq_entries[1,:], marker='x')
#plt.show()

