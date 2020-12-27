'''
  VQVAE utility functions
'''

import numpy as np
from matplotlib import pyplot as plt
import getch

# VQ Pager - plot input/output spectra to sanity check
def vqvae_pager(fig,fs,target,est,worst_fr):
    nb_plots = 8
    w = 0;
    key = ' '
    while key != 'q':
        frames=range(fs,fs+nb_plots)
        nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
        plt.figure(fig)
        plt.clf()
        plt.tight_layout()
        plt.title('Rate K Amplitude Spectra')
        for r in range(nb_plots):
            plt.subplot(nb_plotsy,nb_plotsx,r+1)
            f = frames[r];
            plt.plot(target[f,:],'g')
            plt.plot(est[f,:],'r')
            plt.ylim(0,80)
            a_mse = np.mean((target[f,:]-est[f,:])**2)
            t = "f: %d %3.1f" % (f, a_mse)
            plt.title(t)
        plt.show(block=False)
        plt.pause(0.0001)
        print("n-next b-back w-worst s-save_png q-quit", end='\r', flush=True);
        key = getch.getch()
        if key == 'n':
            fs += nb_plots
        if key == 'b':
            fs -= nb_plots
        if key == 's':
            plt.savefig('vqvae_spectra.png')
        if key == 's':
            plt.savefig('vqvae_spectra.png')
        if key == 'w':
            fs = min(worst_fr[w],target.shape[0]-nb_plots)
            w += 1;
    plt.close()

# Calculate total mean square error and mse per frame
def calc_mse(train, train_est, nb_samples, nb_features, dec):
    msepf = np.zeros(nb_samples-dec)
    e1 = 0; n = 0
    for i in range(nb_samples-dec):
        e = (train_est[i,:] - train[i,:])**2
        msepf[i] = np.mean(e)
        e1 += np.sum(e); n += nb_features
    mse = e1/n
    return mse, msepf

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

# Count how many times each vector is used
def vector_count(x, vq, dim, nb_vecs):
    # VQ search outside of Keras Backend
    flat_inputs = np.reshape(x, (-1, dim))
    distances = np.sum(flat_inputs**2, axis=1, keepdims=True) - 2* np.dot(flat_inputs, vq.T) + np.sum(vq.T ** 2, axis=0, keepdims=True)
    encoding_indices = np.argmax(-distances, axis=1)
    count = np.zeros(nb_vecs, dtype="int")
    count[encoding_indices] += 1
    return count

# use PCA to plot encoder space and VQ in 2D
# https://towardsdatascience.com/principal-component-analysis-pca-from-scratch-in-python-7f3e2a540c51
def find_pca(A):
    # calculate the mean of each column
    M = np.mean(A.T, axis=1)
    # center columns by subtracting column means
    C = A - M
    # calculate covariance matrix of centered matrix
    V = np.cov(C.T)
    # eigendecomposition of covariance matrix
    values, vectors = np.linalg.eig(V)
    P = vectors.T.dot(C.T)
    return P.T

