#!/usr/bin/python3
# eband_out.py
#
# David Rowe Dec 2019
#
# Given rate K vectors and Codec 2 model records containing side
# information such as Wo, output Codec 2 model records using NN for
# rate K -> rate L conversion

'''
  usage: sox -t .sw -r 8000 ~/Downloads/train_8k.sw -t .sw - trim 0 2.5 | c2sim - --bands sample.f32 --modelout sample.model 
         ./eband_out.py ampnn.h5 sample.f32 sample.model --modelout sample_out.model
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
default_eband_K   = 14
N                 = 80

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Decode rate K -> rate L using a NN')
parser.add_argument('ampnn', help='amp NN trained .h5 file')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('modelin', help='Input Codec 2 model records')
parser.add_argument('--modelout', help='Ouput Codec 2 model records with reconstructed rate L vectors')
parser.add_argument('--frames', type=list_str, default="30,31,32,33,34,35", help='Frames to view')
parser.add_argument('--eband_start', type=int, default=0, help='Start element of eband vector')
parser.add_argument('--eband_K', type=int, default=default_eband_K, help='Length of eband vector')
args = parser.parse_args()

eband_K = args.eband_K

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelin)
print(A.shape, Wo.shape)
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

# our model
model = models.Sequential()
model.add(layers.Dense(2*eband_K, activation='relu', input_dim=eband_K))
model.add(layers.Dense(2*eband_K, activation='relu'))
model.add(layers.Dense(width, activation='relu'))
model.add(layers.Dense(width))
model.summary()
model.load_weights(args.ampnn)

# run model
amp_sparse_est = model.predict(rateK)

# extract amplitudes from sparse vector and estimate variance of
# quantisation error (mean error squared between original and
# quantised magnitudes, the spectral distortion)
A_est = np.zeros((nb_samples,codec2_model.max_amp+1))
error = np.zeros(nb_samples)
e1 = 0; n = 0;
for i in range(nb_samples):
    e2 = 0;
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        A_est[i,m] = 10 ** amp_sparse_est[i,bin]
        e = (20*amp_sparse_est[i,bin] - 20*np.log10(A[i,m])) ** 2
        n+=1; e1 += e; e2 += e;
    error[i] = e2/L[i]

# mean of error squared is actually the variance
print("var1: %3.2f var2: %3.2f (dB*dB)" % (e1/n,np.mean(error)))
      
# save to output model file for synthesis
print(A.shape, A_est.shape)
#exit()
if args.modelout:
    codec2_model.write(Wo, L, A_est, phase, voiced, args.modelout)

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

plt.figure(2)
plt.title('Amplitudes Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = int(frames[r]);
    plt.plot(20*np.log10(A[f,1:L[f]+1]),'g')
    plt.plot(20*np.log10(A_est[f,1:L[f]+1]),'r')
    ef = np.var(20*np.log10(A[f,1:L[f]+1]) - 20*np.log10(A_est[f,1:L[f]+1]) )
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
    s_est = sample_time(f, A_est[f,:])
    plt.plot(range(-N,N),s,'g')
    plt.plot(range(-N,N),s_est,'r') 
plt.show(block=False)

plt.figure(4)
plt.title('Histogram of mean error squared per frame')
plt.hist(error,20)
plt.show(block=False)

print("Click on last figure to finish....")
plt.waitforbuttonpress(0)
plt.close()
