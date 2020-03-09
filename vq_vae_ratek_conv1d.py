#!/usr/bin/python3
"""
  From: VQ-VAE_Keras_MNIST_Example.ipynb
  https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

  Trying to perform rate K vector quantisation using a VQ VAE.  4D
  test with simple Dense layers on enc and dec side:

    ~/codec2/build_linux/src/c2sim ~/Downloads/all_speech_8k.sw --bands all_speech_8k.f32 --modelout all_speech_8k.model --bands_lower 1
    ~/codec2/build_linux/misc/extract -s 0 -e 3 -t 14 all_speech_8k.f32 all_speech_8k_0_3.f32
    ./vq_vae_ratek.py all_speech_8k_0_3.f32 --epochs 1 --eband_K 4 --embedding_dim 4

  Note best results obtained with just 1 epoch.

"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt
import argparse

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Dense, Lambda, Reshape, Conv1D, MaxPooling1D, UpSampling1D
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

# Hyper Parameters
train_scale = 0.125
epochs = 20
batch_size = 64
validation_split = 0.1
nb_timesteps = 8

# VQ-VAE Hyper Parameters.
intermediate_dim = 128
embedding_dim =  args.embedding_dim     
num_embeddings = args.num_embedding     
commitment_cost = 0.25                  # Controls the weighting of the loss terms.

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples = int(len(features)/eband_K)
nb_chunks = int(nb_samples/nb_timesteps)
nb_samples = nb_chunks*nb_timesteps
print("nb_samples: %d" % (nb_samples))
train = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))

train_mean = np.mean(train, axis=0)
train -= train_mean
train *= train_scale

print(train.shape)
print(train[0,:])
print(train[nb_timesteps,:])

# reshape into (batch, timesteps, channels) for conv1D
train = train[:nb_samples,:].reshape(nb_chunks, nb_timesteps, eband_K)
print(train.shape)
print(train[0,0,:])
print(train[1,0,:])

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None)

# Encoder
input_shape = (nb_timesteps, nb_features)
inputs = Input(shape=input_shape, name='encoder_input')

x = Conv1D(32, 3, activation='tanh', padding='same')(inputs)
x = MaxPooling1D(pool_size=2, padding='same')(x)
x = Conv1D(16, 3, activation='tanh', padding='same')(x)
enc = MaxPooling1D(pool_size=2, padding='same')(x)

enc_inputs = enc
encoder = Model(inputs, enc)
encoder.summary()

enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc)

# transparent layer (input = output), but stop any weights being changed based on VQ error.  I think.
# Is this how the gradients are copied from the decoder output the decoder input? 
x = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)

x = UpSampling1D(size=2)(x)
x = Conv1D(16, 3, activation='tanh', padding='same')(x)
x = UpSampling1D(size=2)(x)
x = Conv1D(32, 3, activation='tanh', padding='same')(x)
x = Conv1D(eband_K, 3, padding='same')(x)

#x = Dense(embedding_dim, activation='tanh')(x)

# Autoencoder.
vqvae = Model(inputs, x)
vqvae.summary()
data_variance = np.var(train)
loss = vq_vae_loss_wrapper(data_variance, commitment_cost, enc, enc_inputs)
adam = keras.optimizers.Adam(lr=0.0001)
vqvae.compile(loss=loss, optimizer=adam)

plot_model(vqvae, to_file='vq_vae_ratek.png', show_shapes=True)
vq_entries_init = vqvae.get_layer('vqvae').get_weights()[0]

# Callback to plot VQ entries as they evolve
def cb():
    vq_entries = vqvae.get_layer('vqvae').get_weights()[0]
    plt.figure(5)
    plt.clf()
    plt.scatter(vq_entries[0,:],vq_entries[1,:], marker='x')
    plt.draw()
    plt.pause(0.001)
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: cb() )

history = vqvae.fit(train, train,
                    batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split, callbacks = [print_weights])

# back to original shape
train_est = vqvae.predict(train, batch_size=batch_size)
encoder_out = encoder.predict(train, batch_size=batch_size)
train = train.reshape(nb_samples, eband_K)
train_est = train_est.reshape(nb_samples, eband_K)
print(encoder_out.shape)
encoder_out = encoder_out.reshape(nb_chunks*2, embedding_dim)
print(train.shape, nb_samples)

# Plot training results
loss = history.history['loss'] 
val_loss = history.history['val_loss']
num_epochs = range(1, 1 + len(history.history['loss'])) 

plt.figure(1)
plt.plot(num_epochs, loss, label='Training loss')
plt.plot(num_epochs, val_loss, label='Validation loss') 

plt.title('Training and validation loss')
plt.show(block=False)

# Calculate total mean square error and mse per frame

def calc_mse(train, train_est, nb_samples, nb_features, dec):
    msepf = np.zeros(nb_samples-dec)
    e1 = 0; n = 0
    for i in range(nb_samples-dec):
        e = (10*train_est[i,:] - 10*train[i,:])**2
        msepf[i] = np.mean(e)
        e1 += np.sum(e); n += nb_features
    mse = e1/n
    return mse, msepf

mse,msepf = calc_mse(train/train_scale, train_est/train_scale, nb_samples, nb_features, dec)
print("mse: %4.2f dB*dB" % (mse))
plt.figure(2)
plt.plot(msepf)
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
plt.figure(3)
plt.hist(reject_outliers(msepf), bins='fd')

# plot input/output spectra for a few frames to sanity check

nb_plots = 8
frames=range(100,100+nb_plots)
print(frames)
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
plt.figure(4)
plt.tight_layout()
plt.title('Rate K Amplitude Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frames[r];
    plt.plot(10*(train_mean+train[f,:]/train_scale),'g')
    plt.plot(10*(train_mean+train_est[f,:]/train_scale),'r')
    plt.ylim(0,80)
    a_mse = np.mean((10*train[f,:]/train_scale-10*train_est[f,:]/train_scale)**2)
    t = "f: %d %3.1f" % (f, a_mse)
    plt.title(t)
plt.show(block=False)

# plot spaces

plt.figure(5)
plt.scatter(encoder_out[:,0], encoder_out[:,1])
vq_entries = vqvae.get_layer('vqvae').get_weights()[0]
plt.scatter(vq_entries[0,:],vq_entries[1,:], marker='x')
#plt.scatter(vq_entries_init[0,:],vq_entries_init[1,:], marker='o')
plt.show(block=False)
plt.figure(6)
plt.hist2d(encoder_out[:,0],encoder_out[:,1], bins=(50,50))
plt.show(block=False)

plt.waitforbuttonpress(0)
plt.close()
