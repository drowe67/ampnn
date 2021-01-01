#!/usr/bin/python3
# codec2_model.py
# David Rowe Dec 2019
#
# Python Codec 2 model records I/O

import sys, os
import construct
import numpy as np
import matplotlib.pyplot as plt

max_amp = 160
Fs      = 8000
width   = 256
nb_bytes_per_codec2_record = 1300

codec2_model = construct.Struct(
    "Wo" / construct.Float32l,
    "L" / construct.Int32sl,
    "A" / construct.Array(max_amp+1, construct.Float32l),
    "phi" / construct.Array(max_amp+1, construct.Float32l),
    "voiced" / construct.Int32sl
    )

def read(filename, max_nb_samples=1E32):

    nb_samples = int(min(max_nb_samples, os.stat(filename).st_size/nb_bytes_per_codec2_record));
    print("nb_samples: ", nb_samples);
    Wo = np.zeros(nb_samples)
    L = np.zeros(nb_samples, dtype=int)
    A = np.zeros((nb_samples, max_amp+1))
    phi = np.zeros((nb_samples, max_amp+1))
    voiced = np.zeros(nb_samples, dtype=int)

    # Read Codec 2 model records into numpy arrays for further work
    print("reading codec2 records...")
    
    with open(filename, 'rb') as f:
        for i in range(nb_samples):
            model = codec2_model.parse_stream(f)
            Wo[i] = model.Wo
            L[i] = model.L
            A[i,1:L[i]+1] = model.A[1:L[i]+1]
            phi[i,1:L[i]+1] = model.phi[1:L[i]+1]
            voiced[i] = model.voiced
    f.close()

    return Wo, L, A, phi, voiced

def write(Wo, L, A, phi, voiced, filename):
    nb_samples = Wo.size
    with open(filename, 'wb') as f:
        for i in range(nb_samples):
            model = codec2_model.build(dict(Wo=Wo[i], L=L[i], A=A[i,:max_amp+1], phi=phi[i,:max_amp+1], voiced=voiced[i]))
            f.write(model)
            
if __name__ == "__main__":
    # do this first:
    #   /codec2/build_linux$ ./src/c2sim ../raw/hts1a.raw --modelout hts1a.model

    Wo, L, A, phi, voiced = read("/home/david/codec2/build_linux/hts1a.model")
    write(Wo, L, A, phi, voiced, "/home/david/codec2/build_linux/hts1a_out.model")
    
    plt.figure(1)
    plt.plot(Wo*4000/np.pi)
    plt.show(block=False)
    
    plt.figure(2)
    plt.plot(A[30,:])
    plt.show()

