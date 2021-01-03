#!/usr/bin/python3
'''
  rate K vector quantisation of ebands, with rate L {Am} output. 

  Train: 
   $ ./eband_vq_am.py dev-clean-8k.f32 dev-clean-8k-sparse.f32 --epochs 15 --mean_thresh 30 --mean --nb_embedding 2048 --scale 0.02 --nnout 210103_eband_vq_am.npy
  Test:
  $ ~/codec2/build_linux/src/c2sim wav/big_dog.sw --bands big_dog.f32 --modelout big_dog.model
  $ ./eband_vq_am_out.py eband_vq_am.npy big_dog.f32 big_dog.model --modelout big_dog_out.model
  $ ~/codec2/build_linux/src/c2sim wav/big_dog.raw --modelin big_dog_out.model -o big_dog_out.raw

'''

import logging
import os, argparse, getch
import numpy as np
from matplotlib import pyplot as plt

# Give TF "a bit of shoosh" - needs to be placed _before_ "import tensorflow as tf"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from vqvae_models import *
from vqvae_util import *
import codec2_model

# Constants -------------------------------------------------

batch_size       = 64
validation_split = 0.1
nb_timesteps     = 4
width            = 128
max_amp          = 160 
Fs               = 8000
Fcutoff          = 400
N                = 80

# Command line ----------------------------------------------

parser = argparse.ArgumentParser(description='Testing two stage VQ-VAE of rate K vectors, rate L output')
parser.add_argument('nnin', help='model weights and VQs in .npy')
parser.add_argument('featurefile', help='input f32 file of spectral mag vectors, each element is 10*log10(energy), i.e. dB')
parser.add_argument('modelin', help='Input Codec 2 model records')
parser.add_argument('--modelout', help='Output Codec 2 model records with reconstructed rate L vectors')
parser.add_argument('--eband_K', type=int, default=14, help='width of each spectral mag vector')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of spectral mag vectors to train on')
parser.add_argument('--embedding_dim', type=int, default=16,  help='dimension of embedding vectors (VQ dimension)')
parser.add_argument('--nb_embedding', type=int, default=128, help='number of embedded vectors (VQ size)')
parser.add_argument('--scale', type=float, default=0.005,  help='apply this gain to features when read in')
parser.add_argument('--nnout', type=str, default="eband_vq.npy",  help='Name of output NN we have trained (.npy format)')
parser.add_argument('--mean', action='store_true', help='Extract mean from each chunk')
parser.add_argument('--mean_thresh', type=float, default=0.0,  help='Discard chunks with less than this mean threshold')
parser.add_argument('--narrowband', action='store_true', help='weighting function ignores the first and last two bands')
parser.add_argument('--frame', type=int, default="1", help='start frame to view on VQ pager')
parser.add_argument('--plots', action='store_true', help='diagnostic plots and VQ pager')
args = parser.parse_args()
dim = args.embedding_dim
nb_samples = args.nb_samples
eband_K = args.eband_K
nb_features = eband_K
train_scale = args.scale

# read in rate K vectors ---------------------------------------------------------

features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples = int(len(features)/eband_K)
nb_chunks = int(nb_samples/nb_timesteps)
nb_samples = nb_chunks*nb_timesteps
print("rate K nb_samples: %d" % (nb_samples))
features = np.clip(features,0,None); # no crazy low values
features = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))
print("features: ", features.shape);

# read in Codec 2 model file records and set up sparse rate L vectors --------------------

Wo, L, A, phase, voiced = codec2_model.read(args.modelin, nb_samples)

# Avoid harmonics above Fcutoff, as anti-alising filters tend to
# produce very small values that don't affect speech but contribute
# greatly to error
for f in range(nb_samples):
   L[f] = round(L[f]*((Fs/2)-Fcutoff)/(Fs/2))

# set up sparse amp output vectors
print("building sparse output vecs...")
amp_sparse = np.zeros((nb_samples, width+2), dtype='float32')
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp_sparse[i,bin] = 20*np.log10(A[i,m])

# Reshape into "chunks" (batch, nb_timesteps+2, channels) for conv1D.  We need
# timesteps+2 to have a sample either side for conv1d in "valid" mode.

features_chunks = np.zeros((nb_chunks, nb_timesteps+2, nb_features))
features_chunks[0,1:] = features[0:nb_timesteps+1]

mean_chunks = np.zeros(nb_chunks)
mean_orig = np.zeros(nb_chunks)
gmean = np.mean(features[:nb_samples])
if args.mean is False:
    print("global mean: ", gmean)
    
Wo_chunks = np.zeros((nb_chunks, nb_timesteps, 1))

for i in range(1,nb_chunks-1):
    features_chunks[i] = features[i*nb_timesteps-1:(i+1)*nb_timesteps+1]
    mean_orig[i] = np.mean(features_chunks[i])
    if args.mean:
        mean_chunks[i] = mean_orig[i]
    else:
        mean_chunks[i] = gmean
    features_chunks[i] -= mean_chunks[i]
    Wo_chunks[i] = Wo[i*nb_timesteps:(i+1)*nb_timesteps].reshape(nb_timesteps,1);
features_chunks = features_chunks*train_scale    
print("prediction chunks", features_chunks.shape, Wo_chunks.shape)    

# load up model

vqvae,encoder = vqvae_rate_K_L(nb_timesteps, nb_features, dim, args.nb_embedding, width)
vqvae.summary()
with open(args.nnin, 'rb') as f:
    print("reading ", args.nnin)
    vqvae.get_layer("conv1d_a").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("conv1d_b").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("vq1").set_vq(np.load(f, allow_pickle=True))
    vqvae.get_layer("vq2").set_vq(np.load(f, allow_pickle=True))
    vqvae.get_layer("conv1d_c").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("dense1").set_weights(np.load(f, allow_pickle=True))
    vqvae.get_layer("dense2").set_weights(np.load(f, allow_pickle=True))
    w = [np.load(f, allow_pickle=True), np.load(f, allow_pickle=True)]
    vqvae.get_layer("dense3").set_weights(w)

vq_weights = vqvae.get_layer('vq1').get_vq()

print("testing prediction ...")
amp_sparse_chunks_est = vqvae.predict([features_chunks, Wo_chunks], batch_size=batch_size)
encoder_out = encoder.predict(features_chunks, batch_size=batch_size)

# convert output to original shape and dB
for i in range(1,nb_chunks-1):
   amp_sparse_chunks_est[i] = amp_sparse_chunks_est[i]/train_scale + mean_chunks[i]
amp_sparse_est = amp_sparse_chunks_est.reshape(-1, width)
encoder_out = encoder_out.reshape(-1, dim)
nb_samples = amp_sparse_est.shape[0]

# Plot results -------------------------

# Calculate total mean square error and mse per frame

print("measure MSE ...")
msepf = np.zeros(nb_samples)
A_dB = np.zeros((nb_samples,max_amp));
A_est = np.zeros((nb_samples,codec2_model.max_amp+1))
A_est_dB = np.zeros((nb_samples,max_amp));
e1 = 0; n = 0;
for i in range(nb_samples):
    e = 0;
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        A_dB[i,m] = amp_sparse[i,bin]
        A_est_dB[i,m] = amp_sparse_est[i,bin]
        A_est[i,m] = 10 ** (A_est_dB[i,m]/20.0)
        e += (amp_sparse_est[i,bin] - amp_sparse[i,bin]) ** 2
    e1 += e; n += L[i]
    msepf[i] = e/L[i]
mse = e1/n
print("mse: %4.2f dB*dB " % mse)

worst_fr  = np.argsort(-msepf);
worst_mse = np.sort(-msepf);
print(worst_fr[:10], worst_mse[:10]);

# save to output model file for synthesis
if args.modelout:
    codec2_model.write(Wo, L, A_est, phase, voiced, args.modelout)

if args.plots == False:
    quit()
    
plt.figure(1)
plt.plot(reject_outliers(msepf))
plt.title('Spectral Distortion dB*dB per frame')
plt.show(block=False)

plt.figure(2)
plt.title('Histogram of Spectral Distortion dB*dB out to 2*sigma')
plt.hist(reject_outliers(msepf), bins='fd')
plt.show(block=False)

count = np.zeros(args.nb_embedding, dtype="int")
for i in range(0, nb_samples, batch_size):
    count += vector_count(encoder_out[i:i+batch_size], vq_weights, dim, args.nb_embedding)    

plt.figure(3)
plt.plot(count,'bo')
plt.title('Vector Usage Counts for Stage 1')
print(count)
plt.show(block=False)

plt.figure(4)
plt.hist(mean_orig, bins='fd')
plt.title('Mean of each chunk')
plt.show(block=False)
plt.pause(0.0001)

# VQ Pager ----------------------------------------------------

frame = np.array(args.frame,dtype=int)
nb_plots = 8
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
key = ' '
while key != 'q':
    plt.figure(5)
    plt.tight_layout()
    plt.clf()
    plt.title('Amplitudes Spectra')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame+r;
        plt.plot(A_dB[f,1:L[f]+1],'g')
        plt.plot(A_est_dB[f,1:L[f]+1],'r')
        diff = A_dB[f,1:L[f]+1] - A_est_dB[f,1:L[f]+1]
        a_mse = np.sum(diff**2)/L[f]
        t = "f: %d %3.1f" % (f, a_mse)
        plt.title(t)
        plt.ylim(0,80)
    plt.show(block=False)

    plt.pause(0.0001)
    print("n-next b-back x-next50 z-back50 q-quit", end='\r', flush=True);
    key = getch.getch()
    if key == 'n':
        frame += nb_plots
    if key == 'b':
        frame -= nb_plots
    if key == 'x':
        frame += 50
    if key == 'z':
        frame -= 50
   
plt.close()
