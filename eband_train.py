#!/usr/bin/python3
# eband_train.py
#
# David Rowe Dec 2019
#
# Train a NN to model to transform rate K=14 LPCnetstyle eband vectors
# to rate L {Am} samples.  See if we can get better speech quality
# using small dimension vectors that will be easier to quantise.

'''
  usage: ./src/c2sim ~/Downloads/train_8k.sw --modelout ~/phasenn/train_8k.model --bands ~/phasenn/train_8k.f32
         ./eband_train.py train_8k.f32 train_8k.model --epochs 10
'''

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
import codec2_model
import argparse
import os
from keras.layers import Input, Dense, Concatenate
from keras import models,layers
from keras import initializers
from keras import backend as K

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# constants

width             = 128
nb_batch          = 32
default_eband_K   = 14
max_amp           = 160 
nb_plots          = 6
N                 = 80

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to decode eband rate K -> rate L')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('modelfile', help='Codec 2 model records with rate L vectors')
parser.add_argument('--frames', type=list_str, default="30,31,32,33,34,35", help='Frames to view')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_start', type=int, default=0, help='Start element of eband vector')
parser.add_argument('--eband_K', type=int, default=default_eband_K, help='Length of eband vector')
args = parser.parse_args()
assert nb_plots == len(args.frames)

eband_K = args.eband_K

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelfile, args.nb_samples)
nb_samples = Wo.size;
nb_voiced = np.count_nonzero(voiced)
print("nb_samples: %d voiced %d" % (nb_samples, nb_voiced))

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32')
nb_features = default_eband_K
nb_samples1 = len(features)/nb_features
print("nb_samples1: %d" % (nb_samples1))
assert nb_samples == nb_samples1
features = np.reshape(features, (nb_samples, nb_features))
rateK = features[:,args.eband_start:args.eband_start+eband_K]
print(rateK.shape)

# find and subtract mean for each frame
mean_amp = np.zeros(nb_samples)
for i in range(nb_samples):
    mean_amp[i] = np.mean(np.log10(A[i,1:L[i]+1]))

# set up sparse amp output vectors
amp_sparse = np.zeros((nb_samples, width))
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        #amp_sparse[i,bin] = np.log10(A[i,m]) - mean_amp[i]
        amp_sparse[i,bin] = np.log10(A[i,m])

# our model
model = models.Sequential()
model.add(layers.Dense(2*eband_K, activation='relu', input_dim=eband_K))
model.add(layers.Dense(2*eband_K, activation='relu'))
model.add(layers.Dense(width, activation='relu'))
model.add(layers.Dense(width))
model.summary()

# custom loss function - we only care about outputs at the non-zero
# positions in the sparse y_true vector.  To avoid driving the other
# samples to 0 we use a sparse loss function.  The normalisation term
# accounts for the time varying number of non-zero samples per frame.
def sparse_loss(y_true, y_pred):
    mask = K.cast( K.not_equal(y_true, 0), dtype='float32')
    n = K.sum(mask)
    return K.sum(K.square((y_pred - y_true)*mask))/n

# testing custom loss function
y_true = Input(shape=(None,))
y_pred = Input(shape=(None,))
loss_func = K.Function([y_true, y_pred], [sparse_loss(y_true, y_pred)])
assert loss_func([[[0,1,0]], [[2,2,2]]]) == np.array([1])
assert loss_func([[[1,1,0]], [[3,2,2]]]) == np.array([2.5])
assert loss_func([[[0,1,0]], [[0,2,0]]]) == np.array([1])

# fit the model
from keras import optimizers
sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=sparse_loss, optimizer=sgd)
history = model.fit(rateK, amp_sparse, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)

# try model over training database
amp_sparse_est = model.predict(rateK)

# extract amplitudes from sparse vector and estimate variance of
# quantisation error (mean error squared between original and
# quantised magnitudes, the spectral distortion)
amp_est = np.zeros((nb_samples,width))
error = np.zeros(nb_samples)
e1 = 0; n = 0;
for i in range(nb_samples):
    e2 = 0;
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp_est[i,m] = amp_sparse_est[i,bin]
        e = (20*amp_sparse_est[i,bin] - 20*amp_sparse[i,bin]) ** 2
        n+=1; e1 += e; e2 += e;
    error[i] = e2/L[i]
# mean of error squared is actually the variance
print("var1: %3.2f var2: %3.2f (dB*dB)" % (e1/n,np.mean(error)))
      
# synthesise time domain signal
def sample_time(r, A):
    s = np.zeros(2*N);
    for m in range(1,L[r]+1):
        s = s + A[m]*np.cos(m*Wo[r]*range(-N,N) + phase[r,m])
    return s

# plot results

frames = np.array(args.frames,dtype=int)
nb_plots = frames.size
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'valid'], loc='upper right')
plt.title('model loss')
plt.xlabel('epoch')
plt.show(block=False)

plt.figure(2)
plt.title('Amplitudes Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = int(frames[r]);
    plt.plot(20*np.log10(A[f,1:L[f]]),'g')
    plt.plot(20*amp_est[f,1:L[f]],'r')
    ef = np.var(20*np.log10(A[f,1:L[f]])-20*amp_est[f,1:L[f]])
    t = "f: %d %3.1f" % (f, ef)
    plt.title(t)
    plt.ylim(20,80)
plt.show(block=False)

plt.figure(3)
plt.title('Time Domain')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = int(frames[r]);
    s = sample_time(f, A[f,:])
    A_est = 10**(amp_est[f,:])
    s_est = sample_time(f, A_est)
    plt.plot(range(-N,N),s,'g')
    plt.plot(range(-N,N),s_est,'r') 
plt.show(block=False)

plt.figure(4)
plt.title('Histogram of mean error squared per frame')
plt.hist(error,20)
plt.show(block=False)

plt.figure(5)
plt.title('error squared against frame energy')
plt.scatter(mean_amp, error)
plt.show(block=False)

print("Click on last figure to finish....")
plt.waitforbuttonpress(0)
plt.close()
