#!/usr/bin/python3
'''
  rate K vector quantisation using VQ-VAE, kmeans, and conv1

  $ ~/codec2/build_linux/src/c2sim ~/Downloads/dev-clean-8k.sw --bands dev-clean-8k.f32 --bands_lower 1
  $ ./vq_vae_kmeans_conv1d.py dev-clean-8k.f32 

'''

import logging
import os, argparse, getch
import numpy as np
from matplotlib import pyplot as plt

# Give TF "a bit of shoosh" - needs to be placed _before_ "import tensorflow as tf"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from vq_kmeans import *

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
parser.add_argument('--nnout', type=str, default="vqvae_nn.h5", help='Name of output NN we have trained')
parser.add_argument('--mean', action='store_true', help='remove mean of each vector')
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

# normalise
if args.mean:
    # remove mean of every vector
    train_mean = np.mean(features, axis=1)
    print("mean", features.shape,train_mean.shape)
    for i in range(nb_samples):
        features[i,:] -= train_mean[i]
else:
    # remove global mean
    mean = np.mean(features, axis=0);
    features -= mean
    print("mean", mean)
    train_mean = mean*np.ones((nb_samples,eband_K))

features *= train_scale

print("std",np.std(features, axis=0))

# reshape into (batch, timesteps, channels) for conv1D.  We
# concatentate the training material with same sequence of frames at a
# bunch of different time shifts
train = features[:nb_samples,:].reshape(nb_chunks, nb_timesteps+2, eband_K)
print(train.shape)
for i in range(1,nb_timesteps+2):
    features1 = features[i:nb_samples-nb_timesteps-2+i,:].reshape(nb_chunks-1, nb_timesteps+2, eband_K)
    train =  np.concatenate((train,features1))
print(train.shape)
train_target=train[:,1:nb_timesteps+1,:]
print(train_target.shape)

class CustomCallback(tf.keras.callbacks.Callback):
   def on_epoch_begin(self, epoch, logs=None):
       plt.figure(1)
       plt.clf()
       vq_weights = vqvae.get_layer('vq').get_vq()
       plt.scatter(vq_weights[:,0],vq_weights[:,1], marker='.', color="red")
       plt.xlim([-1.5,1.5]); plt.ylim([-1.5,1.5])
       plt.draw()
       plt.pause(0.0001)      
   
# Model --------------------------------------------

x = tf.keras.layers.Input(shape=(nb_timesteps+2, nb_features), name='encoder_input')

# Encoder
z_e = tf.keras.layers.Conv1D(32, 3, activation='tanh', padding='valid', name="conv1d_a")(x)
z_e = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(z_e)
z_e = tf.keras.layers.Conv1D(dim, 3, activation='tanh', padding='same')(z_e)

encoder = tf.keras.Model(x, z_e)
encoder.summary()

# VQ
z_q = VQ_kmeans(dim, args.num_embedding, name="vq")(z_e)
z_q_ = CopyGradient()([z_q, z_e])

# Decoder
p = tf.keras.layers.Conv1D(dim, 3, activation='tanh', padding='same')(z_q_)    
p = tf.keras.layers.UpSampling1D(size=2)(p)
p = tf.keras.layers.Conv1D(32, 3, activation='tanh', padding='same')(p)
p = tf.keras.layers.Conv1D(eband_K, 3, padding='same')(p)

vqvae = tf.keras.Model(x, p)
vqvae.summary()

vqvae.add_loss(commitment_loss(z_e, z_q))
adam = tf.keras.optimizers.Adam(lr=0.0005)
vqvae.compile(loss='mse', optimizer=adam)

# seed VQ - adjusted to avoid Nans when training due to unused VQ entries
vq_initial = np.random.rand(args.num_embedding,dim)*0.1 - 0.05
vqvae.get_layer('vq').set_vq(vq_initial)

history = vqvae.fit(train, train_target, batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,callbacks=[CustomCallback()])

vq_weights = vqvae.get_layer('vq').get_vq().numpy()
           
# Analyse output -----------------------------------------------------------------------

# back to original shape
train_est = vqvae.predict(train, batch_size=batch_size)
encoder_out = encoder.predict(train, batch_size=batch_size)
train_target =  train_target.reshape(-1, eband_K)
train = train.reshape(-1, eband_K)
train_est = train_est.reshape(-1, eband_K)
print(encoder_out.shape)
encoder_out = encoder_out.reshape(-1, dim)
print(train.shape, encoder_out.shape)

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
mse,msepf = calc_mse(train_target/train_scale, train_est/train_scale, nb_samples, nb_features, 1)
print("mse: %4.2f dB*dB" % (mse))
plt.figure(3)
plt.plot(msepf)
plt.title('Spectral Distortion dB*dB per frame')
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
fs = 110;
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
        plt.plot((train_mean[f]+train_target[f,:]/train_scale),'g')
        plt.plot((train_mean[f]+train_est[f,:]/train_scale),'r')
        plt.ylim(0,80)
        a_mse = np.mean((train_target[f,:]/train_scale-train_est[f,:]/train_scale)**2)
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

