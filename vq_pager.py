#!/usr/bin/python3
# vq_pager.py
#
# David Rowe Dec 2019
#
# Plots VQ quantiser entries
'''
  usage: ./vq_pager.py K vq.f32
'''

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy import signal
import codec2_model
import argparse
import os

parser = argparse.ArgumentParser(description='Plot vector entries')
parser.add_argument('K', type=int, help='dimenion of each vector')
parser.add_argument('file1', help='.f32 vqfile')
parser.add_argument('--file2', help='.f32 vqfile')
args = parser.parse_args()

K = args.K

# read in rate K vectors
vq = np.fromfile(args.file1, dtype='float32')
M1 = int(len(vq)/K)
print("file1 entries: %d" % (M1))
vq1 = np.reshape(vq, (M1, K))

if args.file2:
    vq = np.fromfile(args.file2, dtype='float32')
    M2 = int(len(vq)/K)
    print("file2 entries: %d" % (M2))
    vq2 = np.reshape(vq, (M2, K))
    
# plot results

nb_plots = 16
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
offset = 0
loop=True
while loop and (offset < M1):
    plt.figure(1)
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        plt.plot(vq1[offset+r,:],'g')
        if args.file2:
            plt.plot(vq2[offset+r,:],'r')
            
        #plt.ylim(-20,20)
        plt.show(block=False)

    print("[%d .. %d] Any key for next page, click to finish...." % (offset, offset+nb_plots))
    loop = plt.waitforbuttonpress(0)
    plt.clf()
    offset += nb_plots
plt.close()
