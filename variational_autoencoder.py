#!/usr/bin/python3
'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114

  Modified for explerimenstwith rate K vectors:
  usage:

  $ ./variational_autoencoder.py all_speech_8k.f32 --epochs 5 --latent_dim 10

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    sample = z_mean + K.exp(0.5 * z_log_var) * epsilon
    return sample


parser = argparse.ArgumentParser(description='Variational Autoencoder for rate K eband vectors')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_K', type=int, default=14, help='Length of eband vector')
parser.add_argument('--dec', type=int, default=1, help='decimation rate to simulate')
parser.add_argument('--overlap', action='store_true', help='generate more training data with overlapped vectors')
parser.add_argument('--latent_dim', type=int, default=2,  help='dimenion of latent space')
parser.add_argument('--kl_gain', type=float, default=-0.01,  help='dimenion of latent space')
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

'''
train_mean = np.mean(train,axis=0)
train_std = np.std(train,axis=0)
train -= train_mean
train /= train_std
'''

original_dim = nb_features
x_train = train

# network parameters
input_shape = (original_dim, )
intermediate_dim = 512
batch_size = 128
latent_dim = args.latent_dim

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim)(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
plot_model(vae, to_file='vae.png', show_shapes=True)

reconstruction_loss = mse(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= args.kl_gain
vae_loss = K.mean(reconstruction_loss + kl_loss)
#vae_loss = K.mean(reconstruction_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
# train the autoencoder
vae.fit(x_train,
        epochs=args.epochs,
        batch_size=batch_size,
        validation_split=0.1)

# estimate error in dB for model against training database

# TODO add a little noise to encoded words to test robustness of autoencoder
# noise = 0.1*np.random.normal(0,1,(nb_samples, eband_K))
# print(noise.shape, np.std(noise))

test_enc = encoder.predict(train, batch_size=batch_size)
print(test_enc[2].shape)
train_est = decoder.predict(test_enc[2], batch_size=batch_size)
#train_est1 = decoder.predict(np.round(test_enc[2]), batch_size=batch_size)
#print(np.round(test_enc[2])[:5,:])

#train_est = vae.predict(train, batch_size=batch_size)

def calc_var(train, train_est, nb_samples, nb_features, dec):
    mse = np.zeros(nb_samples-dec)
    e1 = 0
    for i in range(nb_samples-dec):
        e = (10*train_est[i,:] - 10*train[i,:])**2
        mse[i] = np.mean(e)
        e1 += np.sum(e)
    var = e1/(nb_samples*nb_features)
    return var
print("var: %4.2f dB*dB" % (calc_var(train, train_est, nb_samples, nb_features, dec)))
#print("var1: %4.2f dB*dB" % (calc_var(train, train_est1, nb_samples, nb_features, dec)))

#print(train[0,:5])
#print(train_est[0,:5])

# two different samples in latent space
'''
print(train[100:101,:])
test_enc=encoder.predict(train[100:101,:])
print(test_enc[2])
test_enc=encoder.predict(train[100:101,:])
print(test_enc[2])
'''

# plot input/ouput spectra

frame=108
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
    #plt.plot(10*train_est1[f,:],'c')
    plt.ylim(0,80)
plt.show(block=False)

# display a 2D plot of the latent space
z_mean, _, _ = encoder.predict(train, batch_size=batch_size)
plt.figure(2)
plt.scatter(z_mean[:, 0], z_mean[:, 1], c='b')
plt.scatter(z_mean[frame:frame+nb_plots, 0], z_mean[frame:frame+nb_plots, 1], c='r')
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.show()
print(np.mean(z_mean, axis=0))
exit()

# plot output spectra over a 2D plane in latent space
plt.figure(2)
plt.tight_layout()
max_x = np.max(z_mean[:, 0]); min_x = np.min(z_mean[:, 0]);
max_y = np.max(z_mean[:, 1]); min_y = np.min(z_mean[:, 1]);
n=6
grid_x = np.linspace(np.round(min_x), np.round(max_x), n)
grid_y = np.linspace(np.round(min_y), np.round(max_y), n)[::-1]
for i, yi in enumerate(grid_y): 
    for j, xi in enumerate(grid_x):
        #print(xi,yi)
        z_sample1 = np.mean(z_mean, axis=0)
        z_sample1[0] = xi; z_sample1[1] = yi;  
        z_sample = np.array([z_sample1])
        x_decoded = decoder.predict(z_sample)
        plt.subplot(n,n,j*n+i+1)
        plt.plot(10*x_decoded[0,:])
        plt.ylim(-20,80)
plt.show()

      
