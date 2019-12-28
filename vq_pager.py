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

parser = argparse.ArgumentParser(description='Plot VQ entries')
parser.add_argument('K', type=int, help='dimenion of eac vector')
parser.add_argument('filename', help='.f32 vqfile')
args = parser.parse_args()

K = args.K

# read in rate K vectors
vq = np.fromfile(args.filename, dtype='float32')
M = int(len(vq)/K)
print("VQ entries: %d" % (M))
vq = np.reshape(vq, (M, K))

# plot results

nb_plots = 16
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
offset = 0
loop=True
while loop and (offset < M):
    plt.figure(1)
    for r in range(nb_plots):
        plt.subplot(nb_plotsy,nb_plotsx,r+1)
        plt.plot(vq[offset+r,:])
        plt.ylim(-20,20)
        plt.show(block=False)

    print("[%d .. %d] Any key for next page, click to finish...." % (offset, offset+nb_plots))
    loop = plt.waitforbuttonpress(0)
    plt.clf()
    offset += nb_plots
plt.close()
