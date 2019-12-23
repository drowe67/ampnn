#!/usr/bin/python3
# rateK_train.py
#
# David Rowe Dec 2019
#
# Experiments in interpolating rate K vectors using NN's and other
# techniques.

'''
  Usage:

  $ c2sim ~/Downloads/train_8k.sw --rateK --rateKout ~/phasenn/rateK.f32
  $ ./rateK_train.py rateK.f32 --dec 4 --frame 30 --epochs 25
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
newamp1_K         = 20
nb_plots          = 6
N                 = 80

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to interpolate rate K vectors')
parser.add_argument('featurefile', help='f32 file of newamp1 rate K vectors')
parser.add_argument('--dec', type=int, default=4, help='decimation rate')
parser.add_argument('--frame', type=int, default="30", help='Frames to view')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
args = parser.parse_args()
dec = args.dec

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32')
nb_features = newamp1_K
nb_samples = int(len(features)/nb_features)
print("nb_samples: %d" % (nb_samples))
rateK = np.reshape(features, (nb_samples, nb_features))/20
print(rateK.shape)

# set up training data
nb_vecs = int(nb_samples/dec)
inputs  = np.zeros((nb_vecs, 2*newamp1_K))
outputs = np.zeros((nb_vecs, 3*newamp1_K))
outputs_lin = np.zeros((nb_vecs, 3*newamp1_K))
outputs_linpf = np.zeros((nb_vecs, 3*newamp1_K))
nv = 0
for i in range(0,nb_samples-dec,dec):
    inputs[nv,:newamp1_K] = rateK[i,:]
    inputs[nv,newamp1_K:] = rateK[i+dec,:]
    for j in range(dec-1):
        st = j*newamp1_K
        outputs[nv,st:st+newamp1_K] = rateK[i+1+j,:]
    # linear interpolation for reference
    c = 1.0/dec; inc = 1.0/dec;
    for j in range(dec-1):
        st = j*newamp1_K
        outputs_lin[nv,st:st+newamp1_K] = (1-c)*inputs[nv,:newamp1_K] + c*inputs[nv,newamp1_K:]
        c += inc
    # linear interpolation with per frame selection of c
    for j in range(dec-1):
        A = inputs[nv,:newamp1_K]; B = inputs[nv,newamp1_K:];
        T = rateK[i+1+j,:]
        c = -np.dot((B-T),(A-B))/np.dot((A-B),(A-B))
        st = j*newamp1_K
        outputs_linpf[nv,st:st+newamp1_K] = c*A + (1-c)*B
    
    nv += 1
print(inputs.shape, outputs.shape)

nn = 1
if nn:
    # our model
    model = models.Sequential()
    model.add(layers.Dense(3*newamp1_K, activation='tanh', input_dim=2*newamp1_K))
    model.add(layers.Dense(3*newamp1_K, activation='tanh'))
    model.add(layers.Dense(3*newamp1_K))
    model.summary()

    # fit the model
    from keras import optimizers
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer="adam")
    history = model.fit(inputs, outputs, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)

    # test the model on the training data
    outputs_nnest = model.predict(inputs)

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'valid'], loc='upper right')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.show(block=False)

# plot results over all frames
var_lin = np.var(20*outputs-20*outputs_lin)
var_linpf = np.var(20*outputs-20*outputs_linpf)
var_nnest = np.var(20*outputs-20*outputs_nnest)
print("var_lin: %3.2f var_linpf: %3.2f var_nnest: %3.2f" % (var_lin, var_linpf, var_nnest))

# plot results for a few frames

nb_plots = dec+1; nb_plotsy = 1; nb_plotsx = nb_plots
frame = int(args.frame/dec)

plt.figure(2)

loop = True
print("Press key to advance, mouse click on last figure to finish....")
while loop:
    plt.title('rate K Amplitude Spectra')
    for d in range(dec+1):
        plt.subplot(1, nb_plots, d+1)
        if d == 0:
            plt.plot(inputs[frame,:newamp1_K],'g')
        elif d == dec:
            plt.plot(inputs[frame,newamp1_K:],'g')
        else: 
            st = (d-1)*newamp1_K
            plt.plot(outputs[frame,st:st+newamp1_K],'g')
            plt.plot(outputs_lin[frame,st:st+newamp1_K],'b')
            if nn:
                plt.plot(outputs_nnest[frame,st:st+newamp1_K],'r')
            else:
                plt.plot(outputs_linpf[frame,st:st+newamp1_K],'r')
        plt.ylim((-1,4))
    var_lin = np.var(20*outputs[frame,:]-20*outputs_lin[frame,:])
    var_linpf = np.var(20*outputs[frame,:]-20*outputs_linpf[frame,:])
    print("frame: %d var_lin: %3.2f " % (frame,var_lin), end='')
    if nn:
        var_nnest = np.var(20*outputs[frame,:]-20*outputs_nnest[frame,:])
        print("var_nnest: %3.2f" % (var_nnest), end='')
    else:
        print("var_linpf: %3.2f" % (var_linpf), end='')
        
    print(flush=True)
    plt.show(block=False)

    loop = plt.waitforbuttonpress(0)
    frame += 1
    plt.clf()
