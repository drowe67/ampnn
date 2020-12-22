#!/usr/bin/python3
'''
  rate K vector quantisation using two stage VQ-VAE, kmeans, and conv1

  1/ Using LPCNet style mel spaced energy bands:
       $ ~/codec2/build_linux/src/c2sim ~/Downloads/dev-clean-8k.sw --bands dev-clean-8k.f32
       $ ./vq_vae_kmeans_conv1d.py dev-clean-8k.f32 --epochs 5 --scale 0.005

  2/ Codec 2 newamp1 K=20 mel spaced bands:
       $ ~/codec2/build_linux/src/c2sim ~/Downloads/dev-clean-8k.sw --rateK --rateKout dev-clean-8k-K20.f32
       $ ./vq_vae_kmeans_conv1d.py dev-clean-8k-K20.f32 --epochs 5 --eband_K 20 --scale 0.005
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
parser.add_argument('featurefile', help='f32 file of spectral mag vectors, each element is log10(energy), i.e. dB/10')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--eband_K', type=int, default=14, help='Length of eband vector')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--embedding_dim', type=int, default=16,  help='dimension of embedding vectors')
parser.add_argument('--num_embedding', type=int, default=2048,  help='number of embedded vectors')
parser.add_argument('--scale', type=float, default=0.125,  help='apply this gain to features when read in')
parser.add_argument('--nnout', type=str, default="", help='Name of output NN we have trained')
parser.add_argument('--mean', action='store_true', help='Extract mean from each chunk')
parser.add_argument('--mean_thresh', type=float, default=0,  help='Discard chunks with less than this mean threshold')
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

# Concatentate the training material with same sequence of frames at a
# bunch of different time shifts, to increase the amount of training material

for i in range(1,nb_timesteps+2):
    features1 = features[i:nb_samples-nb_timesteps-2+i,:].reshape(nb_chunks-1, nb_timesteps+2, eband_K)
    train =  np.concatenate((train,features1))
print(train.shape)
nb_chunks = train.shape[0];
 
# Optional removal of chunks with mean beneath threshold
j = 0;
for i in range(nb_chunks):
    amean = np.mean(train[i])
    if amean > args.mean_thresh:
        train[j] = train[i]
        j += 1
nb_chunks = j        
train = train[:nb_chunks];
print(nb_chunks, train.shape)

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
 
# scale magnitude of training data to get std dev around 1 ish (adjusted by experiment)
train *= train_scale
print("std",np.std(train))

# The target we wish the network to generate is the "inner" nb_timesteps samples
train_target=train[:,1:nb_timesteps+1,:]
print(train_target.shape)

class CustomCallback(tf.keras.callbacks.Callback):
   def on_epoch_begin(self, epoch, logs=None):
       plt.figure(1)
       plt.clf()
       vq1_weights = vqvae.get_layer('vq1').get_vq()
       vq2_weights = vqvae.get_layer('vq2').get_vq()
       plt.scatter(vq1_weights[:,0],vq1_weights[:,1], marker='.', color="red")
       plt.scatter(1+vq2_weights[:,0],1+vq2_weights[:,1], marker='.')
       plt.xlim([-1.5,1.5]); plt.ylim([-1.5,1.5])
       plt.draw()
       plt.pause(0.0001)      
   
# Model --------------------------------------------

vqvae, encoder = vqvae_models(nb_timesteps, nb_features, dim, args.num_embedding)
vqvae.summary()

adam = tf.keras.optimizers.Adam(lr=0.001)
vqvae.compile(loss='mse', optimizer=adam)

# seed VQs
vq_initial = np.random.rand(args.num_embedding,dim)*0.1 - 0.05
vqvae.get_layer('vq1').set_vq(vq_initial)
vqvae.get_layer('vq2').set_vq(vq_initial)

history = vqvae.fit(train, train_target, batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,callbacks=[CustomCallback()])

vq_weights = vqvae.get_layer('vq1').get_vq().numpy()
          
# Analyse output -----------------------------------------------------------------------

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

loss = history.history['loss'] 
val_loss = history.history['val_loss']
num_epochs = range(1, 1 + len(history.history['loss'])) 

plt.figure(2)
plt.plot(num_epochs, loss, label='Training loss')
plt.plot(num_epochs, val_loss, label='Validation loss') 
plt.title('Training and validation loss')
plt.show(block=False)

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

plt.figure(3)
plt.plot(msepf)
plt.title('Spectral Distortion dB*dB per frame')
plt.show(block=False)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
plt.figure(4)
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

plt.figure(5)
plt.plot(count,'bo')
plt.title('Vector Usage Counts for Stage 1')
print(count)
plt.show(block=False)

plt.figure(6)
plt.hist(train_mean, bins='fd')
plt.show(block=False)
plt.title('Mean of each chunk')

# use PCA to plot encoder space and VQ in 2D -----------------------------------------

# https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
def find_pca(A):
    # calculate the mean of each column
    M = np.mean(A.T, axis=1)
    # center columns by subtracting column means
    C = A - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    values, vectors = np.linalg.eig(V)
    #print(vectors)
    #print(values)
    P = vectors.T.dot(C.T)
    return P.T

fig,ax = plt.subplots()
encoder_pca=find_pca(encoder_out)
ax.hist2d(encoder_pca[:,0],encoder_pca[:,1], bins=(50,50))
vq_pca = find_pca(vq_weights)
ax.scatter(vq_pca[:,0],vq_pca[:,1], marker='.', s=4, color="white")
plt.show(block=False)

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
    plt.figure(8)
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

