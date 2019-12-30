#!/usr/bin/python3
# eband_train.py
#
# David Rowe Dec 2019
#
# Autoencoder experiment to map three rate K frames into one.

'''
  usage: ./src/c2sim ~/Downloads/all_speech_8k.sw --bands ~/ampnn/all_speech_8k.f32 --modelout ~/ampnn/all_speech_8k.model 
         ./eband_train.py all_speech_8k.f32 --epochs 25 --dec 3
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
parser.add_argument('--nb_constraint', type=int, default=14, help='Number of frames to train on')
parser.add_argument('--nnout', type=str, default="autonn.h5", help='Name of output NN we have trained')
parser.add_argument('--noplots', action='store_true', help='plot unvoiced frames')
parser.add_argument('--dec', type=int, default=3, help='decimation rate to simulate')
args = parser.parse_args()

eband_K = args.eband_K
dec = args.dec
nb_constraint = args.nb_constraint
nb_features = eband_K*dec

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples = int(len(features)/eband_K)
print("nb_samples: %d" % (nb_samples))
features = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))
print(features.shape)

# bulk up training data using overlapping vectors
train = np.zeros((nb_samples-dec,nb_features))
for i in range(nb_samples-dec):
    for d in range(dec):
        st = d*eband_K
        train[i,st:st+eband_K] = features[i+d,:]
    
# our model
model = models.Sequential()
model.add(layers.Dense(5*nb_features, activation='relu', input_dim=nb_features))
model.add(layers.Dense(nb_constraint, activation='tanh'))
model.add(layers.Dense(5*nb_features, activation='relu', input_dim=nb_features))
model.add(layers.Dense(nb_features))
model.summary()

# fit the model
from keras import optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)
history = model.fit(train, train, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)
model.save(args.nnout)

# try model over training database
train_est = model.predict(train)
mse = np.zeros(nb_samples-dec)
e1 = 0
for i in range(nb_samples-dec):
    e = (10*train_est[i,:] - 10*train[i,:])**2
    mse[i] = np.mean(e)
    e1 += np.sum(e)
var = e1/(nb_samples*nb_features)
print("var: %4.2f dB*dB" % (var))

# plot results

if args.noplots:
    sys.exit(0)

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
plt.hist(reject_outliers(mse), bins='fd')
plt.title('model MSE in dB*dB')
plt.show(block=False)

print("Click on last figure to finish....")
plt.waitforbuttonpress(0)
plt.close()
