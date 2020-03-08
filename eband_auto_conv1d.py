#!/usr/bin/python3
# eband_auto.py
#
# David Rowe Feb 2020
#
# Autoencoder experiment using conv1D to compress rate K eband vectors

'''
  usage:

  ./eband_auto_conv1d.py all_speech_8k.f32
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

nb_batch          = 32
default_eband_K   = 14

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to decode eband rate K -> rate L')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--eband_K', type=int, default=default_eband_K, help='Length of eband vector')
parser.add_argument('--plot_worst', action='store_true', help='plot worst vectors')
parser.add_argument('--plot_random', action='store_true', help='plot set of random vectors')
args = parser.parse_args()

eband_K = args.eband_K
nb_timesteps = 8
train_scale = 0.125

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples = int(len(features)/eband_K)
nb_chunks = int(nb_samples/nb_timesteps)
nb_samples = nb_chunks*nb_timesteps
print("nb_samples: %d" % (nb_samples))
train = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))

train_mean = np.mean(train, axis=0)
train -= train_mean
train *= train_scale

print(train.shape)
print(train[0,:])
print(train[nb_timesteps,:])

# reshape into (batch, timesteps, channels) for conv1D
train = train[:nb_samples,:].reshape(nb_chunks, nb_timesteps, eband_K)
print(train.shape)
print(train[0,0,:])
print(train[1,0,:])
    
# our model
model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', padding='same', input_shape=(nb_timesteps, eband_K)))
model.add(layers.MaxPooling1D(pool_size=2, padding='same'))
model.add(layers.Conv1D(16, 3, activation='relu', padding='same'))
model.add(layers.MaxPooling1D(pool_size=2, padding='same'))

model.add(layers.UpSampling1D(size=2))
model.add(layers.Conv1D(16, 3, activation='relu', padding='same'))
model.add(layers.UpSampling1D(size=2))
model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
model.add(layers.Conv1D(eband_K, 3, padding='same'))
model.summary()
          
from keras import optimizers
model.compile(loss='mse', optimizer='adam')
history = model.fit(train, train, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)

train_est = model.predict(train, batch_size=nb_batch)
train = train.reshape(nb_samples, eband_K)
train_est= train_est.reshape(nb_samples, eband_K)
print(train.shape, nb_samples)

# Plot training results

loss = history.history['loss'] 
val_loss = history.history['val_loss']
num_epochs = range(1, 1 + len(history.history['loss'])) 

plt.figure(1)
plt.plot(num_epochs, loss, label='Training loss')
plt.plot(num_epochs, val_loss, label='Validation loss') 

plt.title('Training and validation loss')
plt.show(block=False)

def calc_mse(train, train_est, nb_samples, nb_features):
    msepf = np.zeros(nb_samples)
    e1 = 0; n = 0
    for i in range(nb_samples):
        e = (10*train_est[i,:] - 10*train[i,:])**2
        msepf[i] = np.mean(e)
        e1 += np.sum(e); n += nb_features
    mse = e1/n
    return mse, msepf
mse,msepf = calc_mse(train/train_scale, train_est/train_scale, nb_samples, eband_K)
print("mse: %4.2f dB*dB" % (mse))

plt.figure(2)
plt.plot(msepf)

plt.figure(3)
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
plt.hist(reject_outliers(msepf), bins='fd')

# plot input/output spectra for a few frames to sanity check

plt.figure(4)
nb_plots = 6
worst_frames = np.argsort(msepf)
one_bad_per_minute = int(nb_samples*2E-4)
print("one bad per minute thresh %d %5.2f dB*dB\n" % (one_bad_per_minute, msepf[worst_frames[-one_bad_per_minute]]))
if args.plot_worst:
    frames = worst_frames[-nb_plots:]
elif args.plot_random:
    frames = np.random.randint(0, nb_samples, nb_plots)
else:
    frames = range(100,100+nb_plots)
print(frames)
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;

plt.tight_layout()
plt.title('Rate K Amplitude Spectra')
for r in range(nb_plots):
    plt.subplot(nb_plotsy,nb_plotsx,r+1)
    f = frames[r];
    plt.plot(10*(train_mean+train[f,:]/train_scale),'g')
    plt.plot(10*(train_mean+train_est[f,:]/train_scale),'r')
    plt.ylim(0,80)
    a_mse = np.mean((10*train[f,:]/train_scale-10*train_est[f,:]/train_scale)**2)
    t = "f: %d %3.1f" % (f, a_mse)
    plt.title(t)
plt.show(block=False)

print("Click on last figure to finish....")
plt.waitforbuttonpress(0)
plt.close()
