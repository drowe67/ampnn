#!/usr/bin/python3
# eband_train.py
#
# David Rowe Dec 2019
#
# Train a NN to model to transform rate K=14 LPCNet style eband vectors
# to rate L {Am} samples.  See if we can get better speech quality
# using small dimension vectors that will be easier to quantise.

'''
  usage: ~/codec2/build_linux/src/c2sim ~/Downloads/dev-clean-8k.sw --bands dev-clean-8k.f32 --modelout dev-clean-8k.model
         ./eband_train.py dev-clean-8k.f32 dev-clean-8k.model --epochs 10 --noplots
'''

import logging
import os, argparse, sys
import numpy as np
from matplotlib import pyplot as plt

# Give TF "a bit of shoosh" - needs to be placed _before_ "import tensorflow as tf"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import codec2_model

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# constants

width             = 128
nb_batch          = 32
default_eband_K   = 14
max_amp           = 160 
nb_plots          = 6
N                 = 80
Fcutoff           = 400

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to decode eband rate K -> rate L')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('modelfile', help='Codec 2 model records with rate L vectors')
parser.add_argument('--frame', type=int, default=1, help='frames to start veiwing')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_start', type=int, default=0, help='Start element of eband vector')
parser.add_argument('--eband_K', type=int, default=default_eband_K, help='Length of eband vector')
parser.add_argument('--nnout', type=str, default="ampnn.h5", help='Name of output NN we have trained')
parser.add_argument('--noplots', action='store_true', help='plot unvoiced frames')
parser.add_argument('--gain', type=float, default=0.1, help='scale factor for eband vectors')
parser.add_argument('--Fs', type=int, default=8000, help='scale factor for eband vectors')
args = parser.parse_args()

eband_K = args.eband_K
Fs = args.Fs

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelfile, args.nb_samples)
nb_samples = Wo.size;
nb_voiced = np.count_nonzero(voiced)
print("nb_samples: %d voiced %d" % (nb_samples, nb_voiced))

# Avoid harmonics above Fcutoff, as anti-alising filters tend to
# produce very small values that don't affect speech but contribute
# greatly to error
for f in range(nb_samples):
   L[f] = round(L[f]*((Fs/2)-Fcutoff)/(Fs/2))
      
# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_features = eband_K
nb_samples1 = len(features)/nb_features
print("nb_samples1: %d" % (nb_samples1))
assert nb_samples == nb_samples1
features = np.reshape(features, (nb_samples, nb_features))
rateK = features[:,args.eband_start:args.eband_start+eband_K]*args.gain
      
# set up sparse amp output vectors
print("building sparse output vecs...")
amp_sparse = np.zeros((nb_samples, width))
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp_sparse[i,bin] = 20*np.log10(A[i,m])*args.gain

print("rateK mean:", np.mean(rateK,axis=1), "std:", np.std(rateK,axis=1))

# our model
model = tf.keras.models.Sequential()
cand=1
if cand == 1:
    model.add(tf.keras.layers.Dense(2*eband_K, activation='relu', input_dim=eband_K))
    model.add(tf.keras.layers.Dense(2*eband_K, activation='relu'))
    model.add(tf.keras.layers.Dense(width, activation='relu'))
    model.add(tf.keras.layers.Dense(width))
else:
    # simple linear model as a control
    model.add(tf.keras.layers.Dense(width,  input_dim=eband_K))
    
model.summary()

# custom loss function - we only care about outputs at the non-zero
# positions in the sparse y_true vector.  To avoid driving the other
# samples to 0 we use a sparse loss function.  The normalisation term
# accounts for the time varying number of non-zero samples per frame.
def sparse_loss(y_true, y_pred):
    mask = tf.cast( tf.math.not_equal(y_true, 0), dtype='float32')
    n = tf.reduce_sum(mask)
    return tf.reduce_sum(tf.math.square((y_pred - y_true)*mask))/n

# fit the model
from keras import optimizers
model.compile(loss=sparse_loss, optimizer='adam')
history = model.fit(rateK, amp_sparse, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)
model.save(args.nnout)

# try model over training database
amp_sparse_est = model.predict(rateK)

# extract amplitudes from sparse vector and estimate 
# quantisation error.  The MSE is the spectral distortion, which
# includes a DC term (fixed gain error).  The variance of the error
# is a better measure of the error in spectral shape.

amp_est = np.zeros((nb_samples,max_amp))
mse = np.zeros(nb_samples)
var = np.zeros(nb_samples)
e1 = 0; n = 0;
for i in range(nb_samples):
    e2 = 0;
    ev = np.zeros(L[i])
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp_est[i,m] = amp_sparse_est[i,bin]
        ev[m-1] = (amp_sparse_est[i,bin] - amp_sparse[i,bin])/args.gain
        e = ev[m-1] ** 2
        e1 += e; e2 += e; n+=1
    mse[i] = e2/L[i]
    var[i] = np.var(ev)
print("mse1: %3.2f mse2: %3.2f var2: %3.2f (dB*dB) " % (e1/n,np.mean(mse),np.mean(var)))
print("%4.2f" % np.mean(var))

# synthesise time domain signal
def sample_time(r, A):
    s = np.zeros(2*N);
    for m in range(1,L[r]+1):
        s = s + A[m]*np.cos(m*Wo[r]*range(-N,N) + phase[r,m])
    return s

# plot results

if args.noplots:
    sys.exit(0)
frame=args.frame
nb_plots = 6
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'valid'], loc='upper right')
plt.title('model loss')
plt.xlabel('epoch')
plt.show(block=False)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

plt.figure(2)
plt.title('Histogram of mean squared error per frame')
plt.hist(reject_outliers(mse), bins='fd')
plt.show(block=False)

plt.figure(3)
plt.plot(mse)
plt.plot(var)

print("Any key to page, click on last figure to finish....")
loop=True
while loop:
    plt.figure(4)
    plt.tight_layout()
    plt.clf()
    plt.title('Amplitudes Spectra')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame + r;
        plt.plot(20*(np.log10(A[f,1:L[f]+1])),'g')
        plt.plot(amp_est[f,1:L[f]+1]/args.gain,'r')
        diff = 20*np.log10(A[f,1:L[f]+1]) - amp_est[f,1:L[f]+1]/args.gain
        a_mse = np.sum(diff**2)/L[f]
        a_var = np.var(diff)
        t = "f: %d %3.1f  %3.1f" % (f, a_mse, a_var)
        plt.title(t)
        plt.ylim(0,70)
    plt.show(block=False)

    plt.figure(5)
    plt.clf()
    plt.title('Time Domain')
    mx = 0
    s = np.zeros((nb_plots, 2*N))
    for r in range(nb_plots):
        f = frame + r;
        s[r,:] = sample_time(f, A[f,:])
        if np.max(np.abs(s)) > mx:
            mx = np.max(np.abs(s))
    mx = 1000*np.ceil(mx/1000)
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame + r;
        A_est = 10**((amp_est[f,:]/args.gain)/20)
        s_est = sample_time(f, A_est)
        plt.plot(range(-N,N),s[r,:],'g')
        plt.plot(range(-N,N),s_est,'r')
        plt.ylim(-mx,mx)
    plt.show(block=False)

    loop = plt.waitforbuttonpress(0)
    frame += nb_plots
plt.close()
