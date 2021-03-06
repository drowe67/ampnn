#!/usr/bin/python3
# ebanddec_train.py
#
# David Rowe Dec 2019
#
# Train a NN to model to transform rate K=14 LPCNet style eband vectors
# to rate L {Am} samples.  See if we can get better speech quality
# using small dimension vectors that will be easier to quantise.
#

# This version incorporates decimation/interpolation.  An undersampled
# (decimated in time) version of rate K vectors is used to synthesise the
# rate L vectors at the orginal frame rate.

'''
  usage: ./src/c2sim ~/Downloads/all_speech_8k.sw --bands ~/ampnn/all_speech_8k.f32 --modelout ~/ampnn/all_speech_8k.model 
         ./ebanddec_train.py all_speech_8k.f32 all_speech_8k.model --epochs 10 --dec 3
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
Fs                = 8000
Fcutoff           = 3600

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to decode eband rate K -> rate L')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('modelfile', help='Codec 2 model records with rate L vectors')
parser.add_argument('--frame', type=int, default=30, help='start frames to view')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_start', type=int, default=0, help='Start element of eband vector')
parser.add_argument('--eband_K', type=int, default=default_eband_K, help='Length of eband vector')
parser.add_argument('--nnout', type=str, default="ampnn.h5", help='Name of output NN we have trained')
parser.add_argument('--noplots', action='store_true', help='plot unvoiced frames')
parser.add_argument('--gain', type=float, default=1.0, help='scale factor for eband vectors')
parser.add_argument('--dec', type=int, default=2, help='decimation rate to simulate')

args = parser.parse_args()

dec = args.dec
assert dec != 1
eband_K = args.eband_K

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelfile, args.nb_samples)
nb_samples = Wo.size;
nb_voiced = np.count_nonzero(voiced)
print("nb_samples: %d voiced %d" % (nb_samples, nb_voiced))

# Avoid harmonics above Fcutoff, as anti-alising filters tend to
# produce very small values that don't affect speech but contribute
# greatly to error
for f in range(nb_samples):
    L[f] = round(L[f]*Fcutoff/(Fs/2))

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_features = eband_K
nb_samples1 = len(features)/nb_features
print("nb_samples1: %d" % (nb_samples1))
assert nb_samples == nb_samples1
features = np.reshape(features, (nb_samples, nb_features))
rateK = features[:,args.eband_start:args.eband_start+eband_K]/args.gain

mean_log10A = np.zeros(nb_samples)
mean_rateK = np.zeros(nb_samples)
for i in range(nb_samples):
    mean_log10A[i] = np.mean(np.log10(A[i,1:L[i]+1]))
    mean_rateK[i] = np.mean(rateK[i,:])
    rateK[i,:] = rateK[i,:] - mean_rateK[i]

# set up sparse amp output vectors
amp_sparse = np.zeros((nb_samples, width))
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp_sparse[i,bin] = np.log10(A[i,m]) - mean_log10A[i]

# reshape to simulate decimation/interpolation, we overlap here to get more training data

rateKdec = np.zeros((nb_samples-dec, 2*eband_K))
for i in range(nb_samples-dec):
    rateKdec[i,:eband_K] = rateK[i,:]
    rateKdec[i,eband_K:] = rateK[i+dec,:]
amp_sparsedec = np.zeros((nb_samples-dec, dec*width))
for i in range(0, nb_samples-dec):
    for d in range(dec):
        st = d*width
        amp_sparsedec[i,st:st+width] = amp_sparse[i+d,:]

# our model
model = models.Sequential()
cand = 2
if cand == 1:
    model.add(layers.Dense(dec*eband_K, activation='relu', input_dim=2*eband_K))
    model.add(layers.Dense(dec*eband_K, activation='relu'))
    model.add(layers.Dense(2*dec*width, activation='relu', input_dim=2*eband_K))
    model.add(layers.Dense(dec*width))

if cand == 2:
    #model.add(layers.Dense(dec*width, activation='relu', input_dim=2*eband_K))
    model.add(layers.Dense(dec*width,  input_dim=2*eband_K))
    model.add(layers.Dense(dec*width))
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
history = model.fit(rateKdec, amp_sparsedec, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)
model.save(args.nnout)

# try model over training database
amp_sparse_est = model.predict(rateKdec)

# Extract amplitudes from sparse vector and estimate error.  Unlike
# training, we step through in dec steps to simulate real world
# operation

amp_est = np.zeros((nb_samples,max_amp))
error = np.zeros(nb_samples)
e1 = 0; n = 0;
for i in range(0,nb_samples-dec,dec):
    for d in range(dec):
        e2 = 0;
        for m in range(1,L[i+d]+1):
            bin = int(np.round(m*Wo[i+d]*width/np.pi)); bin = min(width-1, bin)
            amp_est[i+d,m] = amp_sparse_est[i,bin+d*width]
            e = (20*amp_sparse_est[i,bin+d*width] - 20*amp_sparse[i+d,bin]) ** 2
            n+=1; e1 += e; e2 += e;
        error[i+d] = e2/L[i+d]
# mean of error squared is the variance
print("var1: %3.2f var2: %3.2f (dB*dB)" % (e1/n,np.mean(error)))
print("%4.2f" % (e1/n))
# synthesise time domain signal
def sample_time(r, A):
    s = np.zeros(2*N);
    for m in range(1,L[r]+1):
        s = s + A[m]*np.cos(m*Wo[r]*range(-N,N) + phase[r,m])
    return s

# plot results

if args.noplots:
    sys.exit(0)
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
frame=args.frame

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'valid'], loc='upper right')
plt.title('model loss')
plt.xlabel('epoch')
plt.show(block=False)

plt.figure(2)
plt.title('Histogram of mean error squared per frame')
plt.hist(error,20)
plt.show(block=False)

print("Press any key for next page. Click on last figure to finish....")
loop=True
while loop:
    plt.figure(3)
    plt.clf()
    plt.title('Amplitudes Spectra')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame+r;
        plt.plot(20*np.log10(A[f,1:L[f]+1]),'g')
        plt.plot(20*(amp_est[f,1:L[f]+1]+mean_log10A[f]),'r')
        ef = np.var(20*np.log10(A[f,1:L[f]+1])-20*amp_est[f,1:L[f]+1])
        t = "f: %d %3.1f" % (f, ef)
        plt.title(t)
        plt.ylim(20,80)
    plt.show(block=False)

    plt.figure(4)
    plt.clf()
    plt.title('Time Domain')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame+r;
        s = sample_time(f, A[f,:])
        A_est = 10**(amp_est[f,:]+mean_log10A[f])
        s_est = sample_time(f, A_est)
        plt.plot(range(-N,N),s,'g')
        plt.plot(range(-N,N),s_est,'r') 
    plt.show(block=False)

    loop = plt.waitforbuttonpress(0)
    frame += nb_plots
plt.close()
