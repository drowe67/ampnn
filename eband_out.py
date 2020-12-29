#!/usr/bin/python3
# eband_out.py
#
# David Rowe Dec 2019
#
# Given rate K vectors and Codec 2 model records containing side
# information such as Wo, output Codec 2 model records using NN for
# rate K -> rate L conversion

'''
  usage: ~/codec2/build_linux/src/c2sim wav/big_dog.raw --bands big_dog.f32 --modelout big_dog.model
         ./eband_out.py ampnn.h5 big_dog.f32 big_dog.model --modelout big_dog_out.model
         ~/codec2/build_linux/src/c2sim wav/big_dog.raw --modelin big_dog_out.model -o big_dog_out.raw
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

# constants

width             = 128
default_eband_K   = 14
N                 = 80
Fcutoff           = 400

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Decode rate K -> rate L using a NN')
parser.add_argument('ampnn', help='amp NN trained .h5 file')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('modelin', help='Input Codec 2 model records')
parser.add_argument('--modelout', help='Output Codec 2 model records with reconstructed rate L vectors')
parser.add_argument('--frame', type=int, default="1", help='start frames to view')
parser.add_argument('--eband_start', type=int, default=0, help='Start element of eband vector')
parser.add_argument('--eband_K', type=int, default=default_eband_K, help='Length of eband vector')
parser.add_argument('--noplots', action='store_true', help='plot unvoiced frames')
parser.add_argument('--gain', type=float, default=0.1, help='scale factor for eband vectors')
parser.add_argument('--dec', type=int, default=1, help='input rate K decimation (lin interp)')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--Fs', type=int, default=8000, help='scale factor for eband vectors')
args = parser.parse_args()

eband_K = args.eband_K
Fs = args.Fs

# read in model file records
Wo, L, A, phase, voiced = codec2_model.read(args.modelin, args.nb_samples)
print(A.shape, Wo.shape)
nb_samples = Wo.size;
nb_voiced = np.count_nonzero(voiced)
print("nb_samples: %d voiced %d" % (nb_samples, nb_voiced))

# remove HF, very low amplitude samples that skew results
for f in range(nb_samples):
    L[f] = round(L[f]*((Fs/2)-Fcutoff)/(Fs/2))

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_features = eband_K
nb_samples1 = int(len(features)/nb_features)
features = np.reshape(features, (nb_samples1,nb_features))
if nb_samples > nb_samples1:
    print("warning nb_samples: %d nb_samples1: %d, padding" % (nb_samples, nb_samples1))
    pad=np.zeros((nb_samples - nb_samples1, eband_K))
    print(features.shape, pad.shape)
    features=np.concatenate((features, pad))
print(features.shape)
rateK = features[:,args.eband_start:args.eband_start+eband_K]*args.gain

# optional linear interp/dec
if args.dec != 1:
    dec = args.dec
    inc = 1.0/dec
    print("hello")
    for i in range(0,nb_samples-dec,dec):
        c = 1.0/dec
        left = rateK[i,:]; right = rateK[i+dec,:];
        for d in range(1,dec):
            rateK[i+d,:] = (1-c)*left + c*right
            c += inc
 
        
# our model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2*eband_K, activation='relu', input_dim=eband_K))
model.add(tf.keras.layers.Dense(2*eband_K, activation='relu'))
model.add(tf.keras.layers.Dense(width, activation='relu'))
model.add(tf.keras.layers.Dense(width))
model.summary()
model.load_weights(args.ampnn)

# run model
amp_sparse_est = model.predict(rateK)/args.gain;

# extract amplitudes from sparse vector and estimate 
# quantisation error (mean squared error between original and
# quantised magnitudes, the spectral distortion)
A_est = np.zeros((nb_samples,codec2_model.max_amp+1))
mse = np.zeros(nb_samples)
var = np.zeros(nb_samples)
e1 = 0; n = 0;
for i in range(nb_samples):
    e2 = 0;
    ev = np.zeros(L[i])
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        A_est[i,m] = 10 ** ((amp_sparse_est[i,bin])/20)
        ev[m-1] = amp_sparse_est[i,bin] - 20*np.log10(A[i,m])
        e = ev[m-1] ** 2
        n+=1; e1 += e; e2 += e;
    mse[i] = e2/L[i]
    var[i] = np.var(ev)
print("mse1: %3.2f mse2: %3.2f var2: %3.2f (dB*dB) " % (e1/n,np.mean(mse),np.mean(var)))
print("%4.2f" % np.mean(var))
      
# save to output model file for synthesis
if args.modelout:
    codec2_model.write(Wo, L, A_est, phase, voiced, args.modelout)

# synthesise time domain signal
def sample_time(r, A):
    s = np.zeros(2*N);
    for m in range(1,L[r]+1):
        s = s + A[m]*np.cos(m*Wo[r]*range(-N,N) + phase[r,m])
    return s

# plot results

frame = np.array(args.frame,dtype=int)
nb_plots = 6
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;

if args.noplots:
    sys.exit(0)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

plt.figure(1)
plt.title('Histogram of mean squared error per frame')
plt.hist(reject_outliers(mse), bins='fd')
plt.show(block=False)

plt.figure(2)
plt.plot(mse)
plt.show(block=False)

# ebands:
# 0 200 400 600 800 1k 1.2 1.4 1.6 2k 2.4 2.8 3.2 4k 4.8 5.6 6.8 8k

print("Press any key for next page, click on last figure to finish....")
loop=True
while loop and frame < nb_samples:
    plt.figure(3)
    plt.tight_layout()
    plt.clf()
    plt.title('Amplitudes Spectra')
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        f = frame+r;
        plt.plot(20*np.log10(A[f,1:L[f]+1]),'g')
        plt.plot(20*np.log10(A_est[f,1:L[f]+1]),'r')
        diff = 20*np.log10(A[f,1:L[f]+1]) - 20*np.log10(A_est[f,1:L[f]+1])
        a_mse = np.sum(diff**2)/L[f]
        a_var = np.var(diff)
        t = "f: %d %3.1f  %3.1f" % (f, a_mse, a_var)
        plt.title(t)
        plt.ylim(0,80)
    plt.show(block=False)

    plt.figure(4)
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
        f = frame+r;
        s_est = sample_time(f, A_est[f,:])
        plt.plot(range(-N,N),s[r,:],'g')
        plt.plot(range(-N,N),s_est,'r') 
        plt.ylim(-mx,mx)
    plt.show(block=False)

    loop=plt.waitforbuttonpress(0)
    frame+=nb_plots
   
plt.close()
