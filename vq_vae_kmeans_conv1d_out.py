#!/usr/bin/python3
'''
  Generate output files using rate K vector quantisation using two stage VQ-VAE, kmeans, and conv1.

  Codec 2 newamp1 K=20 mel spaced bands:
       $ sox -t .sw -r 8000 ~/Downloads/train_8k.sw -t .sw - trim 0 2.5 | ~/codec2/build_linux/src/c2sim - --rateK --rateKout test.f32
       $ ./vq_vae_kmeans_conv1d_out.py ampnn.h5 test.f32 test_out.f32 --eband_K 20 --scale 0.005
'''

import logging
import os, argparse, getch
import numpy as np
from matplotlib import pyplot as plt

# Give TF "a bit of shoosh" - needs to be placed _before_ "import tensorflow as tf"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from vqvae_twostage import *

# Constants -------------------------------------------------

batch_size = 64
validation_split = 0.1
nb_timesteps = 4

# Command line ----------------------------------------------

parser = argparse.ArgumentParser(description='Two stage VQ-VAE for rate K vectors')
parser.add_argument('ampnn', help='model weights and VQs in .npy')
parser.add_argument('featurefile', help='input f32 file of spectral mag vectors, each element is 10*log10(energy), i.e. dB')
parser.add_argument('featurefile_out', help='output f32 file of spectral mag vectors, each element is 10*log10(energy), i.e. dB')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_K', type=int, default=14, help='Length of eband vector')
parser.add_argument('--embedding_dim', type=int, default=16,  help='dimension of embedding vectors')
parser.add_argument('--num_embedding', type=int, default=2048,  help='number of embedded vectors')
parser.add_argument('--scale', type=float, default=0.125,  help='apply this gain to features when read in')
parser.add_argument('--mean', action='store_true', help='Extract mean from each chunk')
args = parser.parse_args()
dim = args.embedding_dim
nb_samples = args.nb_samples
eband_K = args.eband_K
nb_features = eband_K
train_scale = args.scale

# read in rate K vectors ---------------------------------------------------------

features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples = int(len(features)/eband_K)
nb_chunks = int(nb_samples/(nb_timesteps+2))
nb_samples = nb_chunks*(nb_timesteps+2)
print("nb_samples: %d" % (nb_samples))
features = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))

# Reshape into "chunks" (batch, nb_timesteps+2, channels) for conv1D.  We need
# timesteps+2 to have a sample either side for conv1d in "valid" mode.

train = features[:nb_samples,:].reshape(nb_chunks, nb_timesteps+2, eband_K)
print(train.shape)

# Optional mean removal of each chunk
if args.mean:
    train_mean = np.zeros(nb_chunks)
    for i in range(nb_chunks):
        train_mean[i] = np.mean(train[i])
        train[i] -= train_mean[i]
else:
    # remove global mean
    mean = np.mean(features)
    train -= mean
    print("mean", mean)
    train_mean = mean*np.ones(nb_chunks)
 
# scale magnitude of training data
train *= train_scale
print("std",np.std(train))

# The target we wish the network to generate is the "inner" nb_timesteps samples
train_target=train[:,1:nb_timesteps+1,:]
print(train_target.shape)

# Model --------------------------------------------

vqvae,encoder = vqvae_models(nb_timesteps, nb_features, dim, args.num_embedding)
vqvae.summary()
with open(args.ampnn, 'rb') as f:
    vqvae.get_layer("conv1d_a").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("conv1d_b").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("vq1").set_vq(np.load(f, allow_pickle=True))
    vqvae.get_layer("vq2").set_vq(np.load(f, allow_pickle=True))
    vqvae.get_layer("conv1d_c").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("conv1d_d").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("conv1d_e").set_weights(np.load(f, allow_pickle=True))

vq_weights = vqvae.get_layer('vq1').get_vq()
train_est = vqvae.predict(train, batch_size=batch_size)
encoder_out = encoder.predict(train, batch_size=batch_size)

# add mean back on for each chunk, and scale back up
for i in range(nb_chunks):
    train_target[i] = train_target[i]/train_scale + train_mean[i]
    train_est[i] = train_est[i]/train_scale + train_mean[i]

# convert chunks back to original shape
train_target = train_target.reshape(-1, eband_K)
train_est = train_est.reshape(-1, eband_K)
encoder_out = encoder_out.reshape(-1, dim)
nb_samples = train_target.shape[0]

# Plot training results -------------------------

# Calculate total mean square error and mse per frame

def calc_mse(train, train_est, nb_samples, nb_features, dec):
    msepf = np.zeros(nb_samples-dec)
    e1 = 0; n = 0
    for i in range(nb_samples-dec):
        e = (train_est[i,:] - train[i,:])**2
        msepf[i] = np.mean(e)
        e1 += np.sum(e); n += nb_features
    mse = e1/n
    return mse, msepf

print("mse",train_target.shape, train_est.shape)
mse,msepf = calc_mse(train_target, train_est, nb_samples, nb_features, 1)
print("mse: %4.2f dB*dB" % (mse))

plt.figure(1)
plt.plot(msepf)
plt.title('Spectral Distortion dB*dB per frame')
plt.show(block=False)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
plt.figure(2)
plt.title('Histogram of Spectral Distortion dB*dB out to 2*sigma')
plt.hist(reject_outliers(msepf), bins='fd')
plt.show(block=False)

# Count how many times each vector is used
def vector_count(x, vq, dim, nb_vecs):
    # VQ search outside of Keras Backend
    flat_inputs = np.reshape(x, (-1, dim))
    distances = np.sum(flat_inputs**2, axis=1, keepdims=True) - 2* np.dot(flat_inputs, vq.T) + np.sum(vq.T ** 2, axis=0, keepdims=True)
    encoding_indices = np.argmax(-distances, axis=1)
    count = np.zeros(nb_vecs, dtype="int")
    count[encoding_indices] += 1
    return count

count = np.zeros(args.num_embedding, dtype="int")
for i in range(0, nb_samples, batch_size):
    count += vector_count(encoder_out[i:i+batch_size], vq_weights, dim, args.num_embedding)    

plt.figure(3)
plt.plot(count,'bo')
plt.title('Vector Usage Counts for Stage 1')
print(count)
plt.show(block=False)

plt.figure(4)
plt.hist(train_mean, bins='fd')
plt.show(block=False)
plt.title('Mean of each chunk')

plt.pause(0.0001)
print("Press any key to start VQ pager....")
key = getch.getch()
plt.close('all')

# VQ Pager - plot input/output spectra to sanity check

nb_plots = 8
fs = 100;
key = ' '
while key != 'q':
    frames=range(fs,fs+nb_plots)
    nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
    plt.figure(5)
    plt.clf()
    plt.tight_layout()
    plt.title('Rate K Amplitude Spectra')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frames[r];
        plt.plot(train_target[f,:],'g')
        plt.plot(train_est[f,:],'r')
        plt.ylim(0,80)
        a_mse = np.mean((train_target[f,:]-train_est[f,:])**2)
        t = "f: %d %3.1f" % (f, a_mse)
        plt.title(t)
    plt.show(block=False)
    plt.pause(0.0001)
    print("n-next b-back s-save_png q-quit", end='\r', flush=True);
    key = getch.getch()
    if key == 'n':
        fs += nb_plots
    if key == 'b':
        fs -= nb_plots
    if key == 's':
        plt.savefig('vqvae_spectra.png')

plt.close()

