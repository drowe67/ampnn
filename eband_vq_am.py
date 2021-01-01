#!/usr/bin/python3
'''
  rate K vector quantisation of ebands, with rate L {Am} output. 

  Integrates vq_vae_kmeans_conv1d.py and ebands_train.py using two stage VQ-VAE, kmeans, and conv1.

  $ ~/codec2/build_linux/src/c2sim ~/Downloads/dev-clean-8k.sw --bands dev-clean-8k.f32 --modelout dev-clean-8k.model
  $ ./model_to_sparse.py dev-clean-8k.model dev-clean-8k-sparse.f32
  $ ./eband_vq_am.py dev-clean-8k.f32 dev-clean-8k-sparse.f32

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

parser = argparse.ArgumentParser(description='Two stage VQ-VAE of rate K vectors, rate L output')
parser.add_argument('featurefile', help='f32 file of spectral mag vectors in dB, e.g. 10*log10(energy)')
parser.add_argument('sparsefile', help='f32 file of sparse spectral mag vectors in dB')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--eband_K', type=int, default=14, help='width of each spectral mag vector')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of spectral mag vectors to train on')
parser.add_argument('--embedding_dim', type=int, default=16,  help='dimension of embedding vectors (VQ dimension)')
parser.add_argument('--nb_embedding', type=int, default=127, help='number of embedded vectors (VQ size)')
parser.add_argument('--scale', type=float, default=0.005,  help='apply this gain to features when read in')
parser.add_argument('--nnout', type=str, help='Name of output NN we have trained (.npy format)')
parser.add_argument('--mean', action='store_true', help='Extract mean from each chunk')
parser.add_argument('--mean_thresh', type=float, default=0.0,  help='Discard chunks with less than this mean threshold')
parser.add_argument('--narrowband', action='store_true', help='weighting function ignores the first and last two bands')
parser.add_argument('--frame', type=int, default="1", help='start frame to view on VQ pager')
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
print("rate K nb_samples: %d" % (nb_samples))
features = np.clip(features,0,None); # no crazy low values
features = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))
print("features: ", features.shape);

# read in sparse records ---------------------------------------------------------

amp_sparse = np.fromfile(args.sparsefile, dtype='float32', count = nb_samples*(width+2))
amp_sparse = amp_sparse.reshape(nb_samples,width+2)
Wo = amp_sparse[:,width]
L = amp_sparse[:,width+1].astype('int')
amp_sparse = amp_sparse[:,:width]

print("amp_sparse:", amp_sparse.shape);

# Reshape into "chunks" (batch, nb_timesteps+2, channels) for conv1D.  We need
# timesteps+2 to have a sample either side for conv1d in "valid" mode.

features_chunks = features[:nb_samples,:].reshape(nb_chunks, nb_timesteps+2, eband_K)
amp_sparse_chunks = amp_sparse[:nb_samples,:].reshape(nb_chunks, nb_timesteps+2, width)
print("reshaped:", features_chunks.shape, amp_sparse_chunks.shape)

# Concatentate the training material with same sequence of frames at a
# bunch of different time shifts, to increase the amount of training material

for i in range(1,nb_timesteps+2):
    features1 = features[i:nb_samples-nb_timesteps-2+i,:].reshape(nb_chunks-1, nb_timesteps+2, eband_K)
    features_chunks =  np.concatenate((features_chunks,features1))
    amp_sparse1 = amp_sparse[i:nb_samples-nb_timesteps-2+i,:].reshape(nb_chunks-1, nb_timesteps+2, width)
    amp_sparse_chunks =  np.concatenate((amp_sparse_chunks, amp_sparse1))
print("concat timeshifts:", features_chunks.shape, amp_sparse_chunks.shape)
nb_chunks = features_chunks.shape[0];


# Optional removal of chunks with mean beneath threshold
j = 0;
mean_orig = np.zeros(nb_chunks)
for i in range(nb_chunks):
    mean_orig[i] = np.mean(features_chunks[i])
    if mean_orig[i] > args.mean_thresh:
        features_chunks[j] = features_chunks[i]
        amp_sparse_chunks[j] = amp_sparse_chunks[i]
        j += 1
nb_chunks = j        
features_chunks = features_chunks[:nb_chunks]
amp_sparse_chunks = amp_sparse_chunks[:nb_chunks]
print("after mean_thresh removal: ", nb_chunks, features_chunks.shape)

'''
# Optional mean removal of each chunk
if args.mean:
    train_mean = np.zeros(nb_chunks)
    for i in range(nb_chunks):
        train_mean[i] = np.mean(features_chunks[i])
        features_train[i] -= train_mean[i]
        amp_sparse_chunks -= train_mean[i]
else:
    # remove global mean
    mean = np.mean(features)
    print("global mean", mean)
    features_chunks -= mean
    amp_sparse_chunks -= mean
    train_mean = mean*np.ones(nb_chunks)
'''
# scale magnitude of training data to get std dev around 1 ish (adjusted by experiment)
features_chunks *= train_scale
amp_sparse_chunks *= train_scale
print("std", np.std(features_chunks), np.std(amp_sparse_chunks))

# The target we wish the network to generate is the "inner" nb_timesteps samples
amp_sparse_chunks_target=amp_sparse_chunks[:,1:nb_timesteps+1,:]
print("target: ", amp_sparse_chunks_target.shape)

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

vqvae,encoder = vqvae_rate_K_L(nb_timesteps, nb_features, dim, args.nb_embedding, width)
vqvae.summary()


'''
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
'''
def sparse_loss(y_true, y_pred):
    mask = tf.cast( tf.math.not_equal(y_true, 0), dtype='float32')
    n = tf.reduce_sum(mask)
    return tf.reduce_sum(tf.math.square((y_pred - y_true)*mask))/n

adam = tf.keras.optimizers.Adam(lr=0.001)
vqvae.compile(loss=sparse_loss, optimizer=adam)

# seed VQs
vq_initial = np.random.rand(args.nb_embedding,dim)*0.1 - 0.05
vqvae.get_layer('vq1').set_vq(vq_initial)
vqvae.get_layer('vq2').set_vq(vq_initial)

history = vqvae.fit(features_chunks, amp_sparse_chunks_target, batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,callbacks=[CustomCallback()])
'''
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
'''

vq_weights = vqvae.get_layer('vq1').get_vq()
          
# Analyse output -----------------------------------------------------------------------

# For testing prediction we need to align with other codec2 information, so arrange in
# sequential overlapping chunks

# subset to save time
nb_samples = min(nb_samples,10000)

nb_chunks = int(nb_samples/nb_timesteps)
features_chunks = np.zeros((nb_chunks, nb_timesteps+2, nb_features));
features_chunks[0,1:] = features[0:nb_timesteps+1]
for i in range(1,nb_chunks-1):
    features_chunks[i] = features[i*nb_timesteps-1:(i+1)*nb_timesteps+1]
features_chunks *= train_scale    
print("features_chunks", features_chunks.shape)    
print("testing prediction ...")

amp_sparse_chunks_est = vqvae.predict(features_chunks, batch_size=batch_size)
encoder_out = encoder.predict(features_chunks, batch_size=batch_size)

# convert output to original shape and dB
amp_sparse_est = amp_sparse_chunks_est.reshape(-1, width)/train_scale
encoder_out = encoder_out.reshape(-1, dim)
nb_samples = amp_sparse_est.shape[0]

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

print("measure MSE ...")
msepf = np.zeros(nb_samples)
A_dB = np.zeros((nb_samples,max_amp));
A_est_dB = np.zeros((nb_samples,max_amp));
e1 = 0; n = 0;
for i in range(nb_samples):
    e = 0;
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        A_dB[i,m] = amp_sparse[i,bin]
        A_est_dB[i,m] = amp_sparse_est[i,bin]
        e += (amp_sparse_est[i,bin] - amp_sparse[i,bin]) ** 2
    e1 += e; n += L[i]
    msepf[i] = e/L[i]
mse = e1/n
print("mse: %4.2f dB*dB " % mse)

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

count = np.zeros(args.nb_embedding, dtype="int")
for i in range(0, nb_samples, batch_size):
    count += vector_count(encoder_out[i:i+batch_size], vq_weights, dim, args.nb_embedding)    

plt.figure(5)
plt.plot(count,'bo')
plt.title('Vector Usage Counts for Stage 1')
print(count)
plt.show(block=False)

plt.figure(6)
plt.hist(mean_orig, bins='fd')
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

# VQ Pager ----------------------------------------------------

frame = np.array(args.frame,dtype=int)
nb_plots = 8
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
while key != 'q':
    plt.figure(7)
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
    print("n-next b-back q-quit", end='\r', flush=True);
    key = getch.getch()
    if key == 'n':
        frame += nb_plots
    if key == 'b':
        frame -= nb_plots
   
plt.close()
