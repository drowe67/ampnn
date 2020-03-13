#!/usr/bin/python3
"""
  From: VQ-VAE_Keras_MNIST_Example.ipynb
  https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

  Simple VQ-VAE Keras example:

    $ ./vq__vae_demo.py

"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt
import argparse

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Dense, Lambda, Subtract, Add
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import LambdaCallback
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
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized1, x_inputs1, quantized2, x_inputs2):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat) / data_variance
        
        e_latent_loss = K.mean((K.stop_gradient(quantized1) - x_inputs1) ** 2)
        q_latent_loss1 = K.mean((quantized1 - K.stop_gradient(x_inputs1)) ** 2)
        q_latent_loss2 = K.mean((quantized2 - K.stop_gradient(x_inputs2)) ** 2)
        loss = q_latent_loss1 + q_latent_loss2 + commitment_cost * e_latent_loss
        
        return recon_loss + loss #* beta
    return vq_vae_loss

# Calculate vq-vae loss.
def vq_vae_loss_wrapper1(data_variance, commitment_cost, quantized1, x_inputs1):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat) / data_variance
        
        e_latent_loss = K.mean((K.stop_gradient(quantized1) - x_inputs1) ** 2)
        q_latent_loss1 = K.mean((quantized1 - K.stop_gradient(x_inputs1)) ** 2)
        loss = q_latent_loss1 + q_latent_loss1 + commitment_cost * e_latent_loss
        
        return recon_loss + loss #* beta
    return vq_vae_loss

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None)

# Callback to plot VQ entries as they evolve
def cb():
    vq1_weights = vqvae.get_layer('vq1').get_weights()[0]
    #vq2_weights = vqvae.get_layer('vq2').get_weights()[0]
    plt.clf()
    plt.scatter(vq1_weights[0,:],vq1_weights[1,:], marker='x')
    #plt.scatter(vq2_weights[0,:],vq2_weights[1,:], marker='x')
    plt.xlim([-3,3]); plt.ylim([-3,3])
    plt.draw()
    plt.pause(0.0001)
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: cb() )

# Hyper Parameters.
batch_size = 64
validation_split = 0.1

parser = argparse.ArgumentParser(description='VQ training test')
parser.add_argument('--test', default="qpsk", help='Test to run')
parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=10000, help='Number of frames to train on')
parser.add_argument('--embedding_dim', type=int, default=2,  help='dimension of embedding vectors')
parser.add_argument('--num_embedding', type=int, default=4,  help='number of embedded vectors')
args = parser.parse_args()
dim = args.embedding_dim
nb_samples = args.nb_samples;

# VQ-VAE Hyper Parameters.
commitment_cost = 0.25 # Controls the weighting of the loss terms.

if args.test == "qpsk":
    # a QPSK constellation with noise
    bits = np.random.randint(2,size=nb_samples*dim).reshape(nb_samples, dim)
    x_train = 2*bits-1 + 0.1*np.random.randn(nb_samples, dim)
if args.test == "gaussian":
    x_train = np.random.randn(nb_samples, dim)
    print("var = %5.2f" % (np.var(x_train)))
    
# Model -------------------------------------

input_shape = (dim, )
x = Input(shape=input_shape, name='encoder_input')
# dummy encoder layer, this would normally be dense/conv
#x = Lambda(lambda inputs: inputs)(input)
x1 = VQVAELayer(dim, args.num_embedding, commitment_cost, name="vq1")(x)
# transparent layer (input = output), but stop any weights being changed based on VQ error
x2 = Lambda(lambda x1: x + K.stop_gradient(x1 - x), name="encoded")(x1)

# Decoder
# just to show where decoder layer goes, will be trained to be indentity
x2 = Dense(2, name='dec_dense')(x2)

# Autoencoder.
vqvae = Model(x, x2)
data_variance = np.var(x_train)
loss = vq_vae_loss_wrapper1(data_variance, commitment_cost, x1, x)
vqvae.compile(loss=loss, optimizer='adam')
vqvae.summary()
plot_model(vqvae, to_file='vq_vae_demo.png', show_shapes=True)

if args.test == "qpsk":
    # seed VQ entries otherwise random start leads to converging on poor VQ extries
    vq = np.array([[1.0,1.0,-1.0,-1.0],[1.0,-1.0,1.0,-1.0]])/10
    vqvae.get_layer('vqvae').set_weights([vq])

history = vqvae.fit(x_train, x_train,
                    batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,
                    callbacks=[esc,print_weights])

x_train_est = vqvae.predict(x_train)
var_start = np.var(x_train)
var_end = np.var(x_train-x_train_est)
nb_bits = np.log2(args.num_embedding);
if nb_bits:
    var_end_theory = var_start*dim/(2**(2*nb_bits))
else:
    var_end_theory = 0
print("nb_bits: %3.1f var_start: %5.4f var_end: %5.4f %5.4f dB" % (nb_bits, var_start, var_end, 10*np.log10(var_end)))
plt.scatter(x_train[:,0],x_train[:,1])
vq1_weights = vqvae.get_layer('vq1').get_weights()[0]
plt.scatter(vq1_weights[0,:],vq1_weights[1,:], marker='x')
plt.show()

