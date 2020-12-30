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
from vqvae_util import *

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
parser.add_argument('--nnout', type=str, help='Name of output NN we have trained')
parser.add_argument('--mean', action='store_true', help='Extract mean from each chunk')
parser.add_argument('--mean_thresh', type=float, default=0.0,  help='Discard chunks with less than this mean threshold')
parser.add_argument('--narrowband', action='store_true', help='weighting function ignores the first and last two bands')
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
features = np.clip(features,0,None); # no crazy low values
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
train_mean_orig = np.zeros(nb_chunks)
for i in range(nb_chunks):
    train_mean_orig[i] = np.mean(train[i])
    if train_mean_orig[i] > args.mean_thresh:
        train[j] = train[i]
        j += 1
nb_chunks = j        
train = train[:nb_chunks];
print("after mean_thresh removal: ", nb_chunks, train.shape)

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

# Plot the VQ space as we train
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

# ability to set up a custom loss function that weights most important parts of speech
w_vec = np.ones(nb_features)
if args.narrowband:
    w_vec[0] = 0; w_vec[1] = 0; w_vec[-2] = 0; w_vec[-1] = 0;
w_timestep = np.zeros((nb_timesteps, nb_features))
w_timestep[:] = w_vec
print(w_timestep)
w = tf.convert_to_tensor(w_timestep, dtype=tf.float32)
def weighted_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(w*(y_pred - y_true)))

adam = tf.keras.optimizers.Adam(lr=0.001)
vqvae.compile(loss=weighted_loss, optimizer=adam)

# seed VQs
vq_initial = np.random.rand(args.num_embedding,dim)*0.1 - 0.05
vqvae.get_layer('vq1').set_vq(vq_initial)
vqvae.get_layer('vq2').set_vq(vq_initial)

history = vqvae.fit(train, train_target, batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,callbacks=[CustomCallback()])

# save_model() doesn't work for me so saving model the hard way ....
if args.nnout is not None:
    with open(args.nnout, 'wb') as f:
        np.save(f, vqvae.get_layer("conv1d_a").get_weights(), allow_pickle=True)
        np.save(f, vqvae.get_layer("conv1d_b").get_weights(), allow_pickle=True)
        np.save(f, vqvae.get_layer("vq1").get_vq(), allow_pickle=True)
        np.save(f, vqvae.get_layer("vq2").get_vq(), allow_pickle=True)
        np.save(f, vqvae.get_layer("conv1d_c").get_weights(), allow_pickle=True)
        np.save(f, vqvae.get_layer("conv1d_d").get_weights(), allow_pickle=True)
        np.save(f, vqvae.get_layer("conv1d_e").get_weights(), allow_pickle=True)

vq_weights = vqvae.get_layer('vq1').get_vq()
          
# Analyse output -----------------------------------------------------------------------

train_est = vqvae.predict(train, batch_size=batch_size)
encoder_out = encoder.predict(train, batch_size=batch_size)

# add mean back on for each chunk, and scale back up
for i in range(nb_chunks):
    train_target[i] = train_target[i]/train_scale + train_mean[i]
    train_est[i] = train_est[i]/train_scale + train_mean[i]

# convert chunks back to original shape
train_target = w_vec*train_target.reshape(-1, eband_K)
train_est = w_vec*train_est.reshape(-1, eband_K)
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

print("mse",train_target.shape, train_est.shape)
mse,msepf = calc_mse(train_target, train_est, nb_samples, nb_features, 1)
print("mse: %4.2f dB*dB" % (mse))
worst_fr  = np.argsort(-msepf);
worst_mse = np.sort(-msepf);
print(worst_fr[:10], worst_mse[:10]);

plt.figure(3)
plt.plot(msepf)
plt.title('Spectral Distortion dB*dB per frame')
plt.show(block=False)

plt.figure(4)
plt.title('Histogram of Spectral Distortion dB*dB out to 2*sigma')
plt.hist(reject_outliers(msepf), bins='fd')
plt.show(block=False)

count = np.zeros(args.num_embedding, dtype="int")
for i in range(0, nb_samples, batch_size):
    count += vector_count(encoder_out[i:i+batch_size], vq_weights, dim, args.num_embedding)    

plt.figure(5)
plt.plot(count,'bo')
plt.title('Vector Usage Counts for Stage 1')
print(count)
plt.show(block=False)

plt.figure(6)
plt.hist(train_mean_orig, bins='fd')
plt.show(block=False)
plt.title('Mean of each chunk')

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

vqvae_pager(8,0,train_target,train_est,worst_fr)
