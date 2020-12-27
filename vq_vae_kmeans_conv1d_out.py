#!/usr/bin/python3
'''
  Generate output files using rate K vector quantisation using two stage VQ-VAE, kmeans, and conv1.

  Codec 2 newamp1 K=20 mel spaced bands:
       $ ~/codec2/build_linux/src/c2sim ~/Downloads/dev-clean-8k.sw --rateK --rateKout dev-clean-8k-K20.f32
       $ ./vq_vae_kmeans_conv1d.py dev-clean-8k-K20.f32 --eband_K 20 --epochs 5 --scale 0.005 --nnout test.npy
         -> 11.48dB*dB
       $ sox -t .sw -r 8000 ~/Downloads/train_8k.sw -t .sw - trim 0 2.5 | ~/codec2/build_linux/src/c2sim - --rateK --rateKout test.f32
       $ ./vq_vae_kmeans_conv1d_out.py test.npy test.f32 --featurefile_out test_out.f32 --eband_K 20 --scale 0.005
         -> 21.6 dB*dB
       $ sox -t .sw -c 1 -r 8000 ~/Downloads/train_8k.sw -t .sw - trim 0 2.5 | ~/codec2/build_linux/src/c2sim - --rateK --rateKin test_out.f32 -o test1.raw

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
from vqvae_util import *

# Constants -------------------------------------------------

batch_size = 64
validation_split = 0.1
nb_timesteps = 4

# Command line ----------------------------------------------

parser = argparse.ArgumentParser(description='Two stage VQ-VAE for rate K vectors')
parser.add_argument('ampnn', help='model weights and VQs in .npy')
parser.add_argument('featurefile', help='input f32 file of spectral mag vectors, each element is 10*log10(energy), i.e. dB')
parser.add_argument('--featurefile_out', help='output f32 file of spectral mag vectors, each element is 10*log10(energy), i.e. dB')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_K', type=int, default=14, help='Length of eband vector')
parser.add_argument('--embedding_dim', type=int, default=16,  help='dimension of embedding vectors')
parser.add_argument('--num_embedding', type=int, default=2048,  help='number of embedded vectors')
parser.add_argument('--scale', type=float, default=0.125,  help='apply this gain to features when read in')
parser.add_argument('--mean', action='store_true', help='Extract mean from each chunk')
parser.add_argument('--mean_thresh', type=float, default=0.0,  help='Lower limit of frame mean')
parser.add_argument('--plots', action='store_true', help='diagnostic plots and VQ pager')
args = parser.parse_args()
dim = args.embedding_dim
nb_samples = args.nb_samples
eband_K = args.eband_K
nb_features = eband_K
train_scale = args.scale

# read in rate K vectors ---------------------------------------------------------

features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples_file = int(len(features)/eband_K)
nb_chunks = int(nb_samples_file/nb_timesteps)
nb_samples = nb_chunks*(nb_timesteps)
print("nb_samples:", nb_samples, "nb_chunks:", nb_chunks)

target = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))

# Optional lower limit mean 
for i in range(nb_samples):
    amean = np.mean(target[i])
    if amean < args.mean_thresh:
        target[i] += args.mean_thresh - amean;
print(target[0])

# Reshape into "chunks" (batch, nb_timesteps+2, channels) for conv1D.  We need
# timesteps+2 to have a sample either side for conv1d in "valid" mode.

target_padded = np.zeros((nb_chunks, nb_timesteps+2, nb_features));
target_padded[0,1:] = target[0:nb_timesteps+1]
for i in range(1,nb_chunks-1):
    target_padded[i] = target[i*nb_timesteps-1:(i+1)*nb_timesteps+1]
print("target", target.shape, target_padded.shape)    

# Optional mean removal of each chunk
if args.mean:
    target_mean = np.zeros(nb_chunks)
    for i in range(nb_chunks):
        target_mean[i] = np.mean(target_padded[i])
        target_padded[i] -= target_mean[i]
else:
    # remove global mean
    mean = np.mean(features)
    target_padded -= mean
    print("mean", mean)
    target_mean = mean*np.ones(nb_chunks)

# scale magnitude of training data
target_padded *= train_scale
print("std",np.std(target_padded))

# Build model and load weights and VQ

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
target_est = vqvae.predict(target_padded, batch_size=batch_size)
encoder_out = encoder.predict(target_padded, batch_size=batch_size)

# add mean back on for each chunk, and scale back up
for i in range(nb_chunks):
    target_est[i] = target_est[i]/train_scale + target_mean[i]

# convert output chunks back to original shape
target_est = target_est.reshape(-1, eband_K)
encoder_out = encoder_out.reshape(-1, dim)
print("target_est", target_est.shape, nb_samples)

# make output file the same size despite chunking
if args.featurefile_out is not None:
    target_est = np.concatenate((target_est, np.zeros((nb_samples_file-nb_samples,nb_features))))
    print(target_est.shape, nb_samples_file)
    target_est_out = target_est.astype(np.float32);
    print(features.shape, target_est_out.shape, target_est_out.dtype);
    target_est_out.tofile(args.featurefile_out)

# Plot training results -------------------------

# Calculate total mean square error and mse per frame

print("mse",target.shape, target_est.shape)
mse,msepf = calc_mse(target, target_est, nb_samples, nb_features, 1)
print("mse: %4.2f dB*dB" % (mse))
worst_fr  = np.argsort(-msepf);
worst_mse = np.sort(-msepf);
print(worst_fr[:10], worst_mse[:10]);
if args.plots == False:
    quit();
    
plt.figure(1)
plt.plot(msepf)
plt.title('Spectral Distortion dB*dB per frame')
plt.show(block=False)

plt.figure(2)
plt.title('Histogram of Spectral Distortion dB*dB out to 2*sigma')
plt.hist(reject_outliers(msepf), bins='fd')
plt.show(block=False)

count = np.zeros(args.num_embedding, dtype="int")
for i in range(0, nb_samples, batch_size):
    count += vector_count(encoder_out[i:i+batch_size], vq_weights, dim, args.num_embedding)    

plt.figure(3)
plt.plot(count,'bo')
plt.title('Vector Usage Counts for Stage 1')
print(count)
plt.show(block=False)

plt.figure(4)
plt.hist(target_mean, bins='fd')
plt.show(block=False)
plt.title('Mean of each chunk')

vqvae_pager(5,0,target,target_est,worst_fr)

