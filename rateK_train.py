#!/usr/bin/python3
# rateK_train.py
#
# David Rowe Dec 2019
#
# Experiments in interpolating rate K vectors using NN's and other
# techniques.

'''
  Usage:

  $ ~/codec2/build_linux/src/c2sim ~/Downloads/all_speech_8k.sw --bands all_speech_8k.f32 
  $ ./rateK_train.py all_speech_8k.f32 --dec 3
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
nb_plots          = 6
N                 = 80

def list_str(values):
    return values.split(',')

parser = argparse.ArgumentParser(description='Train a NN to interpolate rate K vectors')
parser.add_argument('featurefile', help='f32 file of rate K vectors')
parser.add_argument('--eband_K', type=int, default=default_eband_K, help='Length of eband vector')
parser.add_argument('--dec', type=int, default=3, help='decimation rate')
parser.add_argument('--frame', type=int, default="30", help='Frames to view')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--nn', action='store_true', help='Run interpolation NN')
parser.add_argument('--nnpf', action='store_true', help='Run pf coeff NN')
args = parser.parse_args()
dec = args.dec
eband_K = args.eband_K

# read in rate K vectors
features = np.fromfile(args.featurefile, dtype='float32')
nb_features = eband_K
nb_samples = int(len(features)/nb_features)
print("nb_samples: %d" % (nb_samples))
rateK = np.reshape(features, (nb_samples, nb_features))
print(rateK.shape)

# set up training data
nb_vecs = int(nb_samples/dec)
print("nb_samples: %d nb_vecs: %d dec: %d" % (nb_samples, nb_vecs, dec))
inputs  = np.zeros((nb_vecs, 2*eband_K))
outputs = np.zeros((nb_vecs, (dec-1)*eband_K))
outputs_lin = np.zeros((nb_vecs, (dec-1)*eband_K))
outputs_linpf = np.zeros((nb_vecs, (dec-1)*eband_K))
output_c = np.zeros((nb_vecs, dec-1))
outputs_step = np.zeros((nb_vecs, (dec-1)*eband_K))
outputs_linstep = np.zeros((nb_vecs, (dec-1)*eband_K))

for i in range(nb_vecs-dec):
    j = i*dec
    inputs[i,:eband_K] = rateK[j,:]
    inputs[i,eband_K:] = rateK[j+dec,:]
    # target outputs
    for d in range(dec-1):
        st = d*eband_K
        outputs[i,st:st+eband_K] = rateK[j+d+1,:]
    # linear interpolation for reference
    c = 1.0/dec; inc = 1.0/dec;
    for d in range(dec-1):
        st = d*eband_K
        outputs_lin[i,st:st+eband_K] = (1-c)*inputs[i,:eband_K] + c*inputs[i,eband_K:]
        c += inc
    # linear interpolation with per frame selection of c
    A = inputs[i,:eband_K]; B = inputs[i,eband_K:];
    for d in range(dec-1):
        T = rateK[j+d+1,:]
        c = -np.dot((B-T),(A-B))/np.dot((A-B),(A-B))
        st = d*eband_K
        outputs_linpf[i,st:st+eband_K] = c*A + (1-c)*B
        output_c[i,d] = c

    # linear interpolation/step function (requires a bit)
    for d in range(dec-1):
        st = d*eband_K
        outputs_step[i,st:st+eband_K] = inputs[i,eband_K:]
    # determine linear interp error
    e_lin = np.var(outputs[i,:] - outputs_lin[i,:])    
    # determine step function error
    e_step = np.var(outputs[i,:] - outputs_step[i,:])    
    if e_lin < e_step:
        outputs_linstep[i,:] = outputs_lin[i,:]
    else:
        outputs_linstep[i,:] = outputs_step[i,:]
        
if args.nn:
    print(inputs.shape, outputs.shape)

    # our model
    model = models.Sequential()
    model.add(layers.Dense(3*eband_K, activation='relu', input_dim=2*eband_K))
    model.add(layers.Dense(3*eband_K, activation='relu'))
    model.add(layers.Dense((dec-1)*eband_K))
    model.summary()

    # fit the model
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
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

# attempt to estimate pf coeffcients from vectors
if args.nnpf:
    print(inputs.shape, output_c.shape)
    print(output_c[:10])
    # our model
    model = models.Sequential()
    model.add(layers.Dense(3*eband_K, activation='relu', input_dim=2*eband_K))
    model.add(layers.Dense(3*eband_K, activation='relu'))
    model.add(layers.Dense(dec-1))
    model.summary()

    # fit the model
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer=sgd)
    history = model.fit(inputs, output_c, batch_size=nb_batch, epochs=args.epochs, validation_split=0.1)
    
    # test the model on the training data
    c_est = model.predict(inputs)

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'valid'], loc='upper right')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.show(block=False)

# plot results over all frames, ebands are log10(energies) so multiply
# by 10 to get variance in dB*dB
var_lin = np.var(10*outputs-10*outputs_lin)
var_linpf = np.var(10*outputs-10*outputs_linpf)
var_linstep = np.var(10*outputs-10*outputs_linstep)
if args.nn:
    var_nnest = np.var(10*outputs-10*outputs_nnest)
    print("var_lin: %3.2f var_linpf: %3.2f var_nnest: %3.2f" % (var_lin, var_linpf, var_nnest))
elif args.nnpf:
    # linear interpolation with per frame selection of c
    outputs_linpf_est = np.zeros((nb_vecs, (dec-1)*eband_K))
    for i in range(nb_vecs-dec):
        A = inputs[i,:eband_K]; B = inputs[i,eband_K:];
        for d in range(dec-1):
            c = c_est[i,d]
            st = d*eband_K
            outputs_linpf_est[i,st:st+eband_K] = c*A + (1-c)*B
    var_nnpfest = np.var(10*outputs-10*outputs_linpf_est)
    print("var_lin: %3.2f var_linpf: %3.2f var_nnpfest: %3.2f" % (var_lin, var_linpf, var_nnpfest))
else:
    print("var_lin: %3.2f var_linpf: %3.2f var_linstep: %3.2f" % (var_lin, var_linpf, var_linstep))

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
            plt.plot(inputs[frame,:eband_K],'g')
        elif d == dec:
            plt.plot(inputs[frame,eband_K:],'g')
        else: 
            st = (d-1)*eband_K
            plt.plot(outputs[frame,st:st+eband_K],'g')
            plt.plot(outputs_lin[frame,st:st+eband_K],'b')
            plt.plot(outputs_linstep[frame,st:st+eband_K],'c')
            if args.nn:
                plt.plot(outputs_nnest[frame,st:st+eband_K],'r')
            elif args.nnpf:
                plt.plot(outputs_linpf_est[frame,st:st+eband_K],'r')
            else:
                plt.plot(outputs_linpf[frame,st:st+eband_K],'r')
        plt.ylim((0,10))
    var_lin = np.var(10*outputs[frame,:]-10*outputs_lin[frame,:])
    var_linpf = np.var(10*outputs[frame,:]-10*outputs_linpf[frame,:])
    var_linstep = np.var(10*outputs[frame,:]-10*outputs_linstep[frame,:])
    print("frame: %d var_lin (b): %3.2f " % (frame,var_lin), end='')
    if args.nn:
        var_nnest = np.var(10*outputs[frame,:]-10*outputs_nnest[frame,:])
        print("var_nnest(r): %3.2f" % (var_nnest), end='')
    elif args.nnpf:
        var_nnpfest = np.var(10*outputs[frame,:]-10*outputs_linpf_est[frame,:])
        print("var_nnpfest(r): %3.2f" % (var_nnpfest), end='')
    else:
        print("var_linpf(r): %3.2f var_linstep(c): %3.2f" % (var_linpf, var_linstep), end='')
        
    print(flush=True)
    plt.show(block=False)

    loop = plt.waitforbuttonpress(0)
    frame += 1
    plt.clf()
