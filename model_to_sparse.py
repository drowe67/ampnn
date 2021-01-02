#!/usr/bin/python3
'''
  Pre-processing step to generate sparse vectors for input to eband_vq_am.py 

  $ ~/codec2/build_linux/src/c2sim ~/Downloads/dev-clean-8k.sw --bands dev-clean-8k.f32 --modelout dev-clean-8k.model
  $ ./model_to_sparse.py dev-clean-8k.model dev-clean-8k-sparse.f32

'''

import os, argparse
import numpy as np
from matplotlib import pyplot as plt

import codec2_model

# Constants -------------------------------------------------

max_amp          = 160 
Fs               = 8000
Fcutoff          = 400
width            = 128

# Command line ----------------------------------------------

parser = argparse.ArgumentParser(description='Generate sparse training vectors')
parser.add_argument('modelfile', help='Codec 2 model records with rate L vectors')
parser.add_argument('sparsefile', help='f32 file of sparse spectral mag vectors in dB')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of spectral mag vectors to train on')

args = parser.parse_args()
nb_samples = args.nb_samples

# read in Codec 2 model file records and set up sparse rate L vectors --------------------

Wo, L, A, phase, voiced = codec2_model.read(args.modelfile, args.nb_samples)
nb_samples = Wo.shape[0]

# Avoid harmonics above Fcutoff, as anti-alising filters tend to
# produce very small values that don't affect speech but contribute
# greatly to error
for f in range(nb_samples):
   L[f] = round(L[f]*((Fs/2)-Fcutoff)/(Fs/2))

# set up sparse amp output vectors
print("building sparse output vecs...")
amp_sparse = np.zeros((nb_samples, width+2), dtype='float32')
for i in range(nb_samples):
    for m in range(1,L[i]+1):
        bin = int(np.round(m*Wo[i]*width/np.pi)); bin = min(width-1, bin)
        amp_sparse[i,bin] = 20*np.log10(A[i,m])

amp_sparse[:,width] = Wo
amp_sparse[:,width+1] = L
print("amp_sparse:", amp_sparse.shape);
amp_sparse.tofile(args.sparsefile)
