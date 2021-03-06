#!/usr/bin/python3
"""
  From: VQ-VAE_Keras_MNIST_Example.ipynb
  https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

  Simple VQ-VAE Keras demo:

     $ ./vq_vae_demo.py
"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras import Model
tf.compat.v1.disable_eager_execution()

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# VQ layer.
class VQVAELayer(tf.keras.layers.Layer):
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
        flat_inputs = tf.reshape(x, (-1, self.embedding_dim))
        
        # Calculate distances of input to embedding vectors.
        distances = (tf.math.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * tf.tensordot(flat_inputs, self.w, 1)
                     + tf.math.reduce_sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = tf.argmax(-distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = tf.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices)

# Calculate vq-vae loss.
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = tf.losses.mse(x, x_hat) / data_variance
        
        e_latent_loss = tf.math.reduce_mean((tf.stop_gradient(quantized) - x_inputs) ** 2)
        q_latent_loss = tf.math.reduce_mean((quantized - tf.stop_gradient(x_inputs)) ** 2)
        loss = q_latent_loss + commitment_cost * e_latent_loss
        
        return recon_loss + loss #* beta
    return vq_vae_loss


# EarlyStoppingCallback.
esc = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None)

# Callback to plot VQ entries as they evolve
def cb():
    vq_entries = vqvae.get_layer('vqvae').get_weights()[0]
    plt.clf()
    plt.scatter(vq_entries[0,:],vq_entries[1,:], marker='x')
    plt.xlim([-3,3]); plt.ylim([-3,3])
    plt.draw()
    plt.pause(0.0001)
print_weights = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: cb() )


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
    
# Encoder
input_shape = (dim, )
inputs = Input(shape=input_shape, name='encoder_input')
# dummy encoder layer, this would normally be dense/conv
enc = Lambda(lambda inputs: inputs, name="lambda1")(inputs)
enc_inputs = enc
enc = VQVAELayer(dim, args.num_embedding, commitment_cost, name="vqvae")(enc)
# transparent layer (input = output), but stop any weights being changed based on VQ error
x = Lambda(lambda enc: enc_inputs + tf.stop_gradient(enc - enc_inputs), name="encoded")(enc)

# Decoder
# just to show where decoder layer goes, will be trained to be identity
x = Dense(2, name='dec_dense')(x)

# Autoencoder.
vqvae = Model(inputs, x)
data_variance = np.var(x_train)
loss = vq_vae_loss_wrapper(data_variance, commitment_cost, enc, enc_inputs)
vqvae.compile(loss=loss, optimizer='adam')
vqvae.summary()
tf.keras.utils.plot_model(vqvae, to_file='vq_vae_demo.png', show_shapes=True)

if args.test == "qpsk":
    # seed VQ entries otherwise random start leads to converging on poor VQ extries
    vq = np.array([[1.0,1.0,-1.0,-1.0],[1.0,-1.0,1.0,-1.0]])/10
    vqvae.get_layer('vqvae').set_weights([vq])

history = vqvae.fit(x_train, x_train,
                    batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,
                    callbacks=[esc,print_weights])
vq_entries = vqvae.get_layer('vqvae').get_weights()[0]

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
plt.scatter(vq_entries[0,:],vq_entries[1,:], marker='x')
#print(vqvae.get_layer('dec_dense').get_weights()[0])
plt.show()

