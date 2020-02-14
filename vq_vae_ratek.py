#!/usr/bin/python3
"""
  From: VQ-VAE_Keras_MNIST_Example.ipynb
  https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

  Trying to perform rate K vector quantisation using a VQ VAE
"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt
import argparse

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Dense, Lambda, Reshape
from keras import losses
from keras import backend as K
from keras.utils import plot_model

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
        
        return recon_loss + loss
    return vq_vae_loss

parser = argparse.ArgumentParser(description='Variational Autoencoder for rate K eband vectors')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_K', type=int, default=14, help='Length of eband vector')
parser.add_argument('--dec', type=int, default=1, help='decimation rate to simulate')
parser.add_argument('--overlap', action='store_true', help='generate more training data with overlapped vectors')
parser.add_argument('--embedding_dim', type=int, default=14,  help='dimension of embedding vectors')
parser.add_argument('--num_embedding', type=int, default=128,  help='number of embedded vectors')
args = parser.parse_args()
eband_K = args.eband_K
dec = args.dec
nb_features = eband_K*dec

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples = int(len(features)/eband_K)
print("nb_samples: %d" % (nb_samples))
features = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))
print(features.shape)

# bulk up training data using overlapping vectors
if args.overlap:
    train = np.zeros((nb_samples-dec,nb_features))
    for i in range(nb_samples-dec):
        for d in range(dec):
            st = d*eband_K
            train[i,st:st+eband_K] = features[i+d,:]
else:
    print(features.shape)
    nb_samples = int(nb_samples/dec)
    print(nb_samples)
    train = features[:nb_samples*dec,:].reshape((nb_samples, nb_features))

# Hyper Parameters.
epochs = 20
batch_size = 64
validation_split = 0.1

# VQ-VAE Hyper Parameters.
intermediate_dim = 512
embedding_dim =  args.embedding_dim     
num_embeddings = args.num_embedding     
commitment_cost = 0.25                  # Controls the weighting of the loss terms.

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None)

# Encoder
input_shape = (nb_features, )
inputs = Input(shape=input_shape, name='encoder_input')
# dummy encoder layer, this would normally be dense/conv
enc = Dense(intermediate_dim, activation='relu')(inputs)
enc = Dense(embedding_dim, activation='relu')(enc)
enc_inputs = enc
enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc)
# transparent layer (input = output), but stop any weights being changed based on VQ error.  I think
x = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)

# Decoder
x = Dense(intermediate_dim, activation='relu')(x)
x = Dense(eband_K)(x)

# Autoencoder.
vqvae = Model(inputs, x)
data_variance = np.var(train)
loss = vq_vae_loss_wrapper(data_variance, commitment_cost, enc, enc_inputs)
vqvae.compile(loss=loss, optimizer='adam')
vqvae.summary()
plot_model(vqvae, to_file='vq_vae_ratek.png', show_shapes=True)

history = vqvae.fit(train, train,
                    batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,
                    callbacks=[esc])

def calc_var(train, train_est, nb_samples, nb_features, dec):
    mse = np.zeros(nb_samples-dec)
    e1 = 0
    for i in range(nb_samples-dec):
        e = (10*train_est[i,:] - 10*train[i,:])**2
        mse[i] = np.mean(e)
        e1 += np.sum(e)
    var = e1/(nb_samples*nb_features)
    return var

train_est = vqvae.predict(train, batch_size=batch_size)
print("var: %4.2f dB*dB" % (calc_var(train, train_est, nb_samples, nb_features, dec)))

# plot input/output spectra

frame=100
nb_plots = 8
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
plt.figure(1)
plt.tight_layout()
plt.title('Rate K Amplitude Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frame+r;
    plt.plot(10*train[f,:],'g')
    plt.plot(10*train_est[f,:],'r')
    plt.ylim(0,80)
plt.show()
