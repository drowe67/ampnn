#!/usr/bin/python3
"""

  Two stage rate K vector quantisation using a VQ VAE and conv1D:

    ~/codec2/build_linux/src/c2sim ~/Downloads/all_speech_8k.sw --bands all_speech_8k.f32 --modelout all_speech_8k.model --bands_lower 1
    ./vq_vae_conv1d_2stage.py all_speech_8k.f32 --embedding_dim 16 --epochs 25 --num_embedding 2048

  -> 6.52 dB*dB   

  [1] VQ-VAE_Keras_MNIST_Example.ipynb
      https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb
"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt
import argparse

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Dense, Lambda, Subtract, Add, Reshape, Conv1D, MaxPooling1D, UpSampling1D
from keras import losses
from keras import backend as K
from keras.utils import plot_model
from keras.callbacks import LambdaCallback
import os

# less verbose tensorflow ....
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# VQ layer.
class VQVAELayer(Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                  shape=(self.embedding_dim, self.num_embeddings),
                                  initializer=self.initializer,
                                  trainable=True)
        
        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))
        
        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_embeddings)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        # Metrics.
        #avg_probs = K.mean(encodings, axis=0)
        #perplexity = K.exp(- K.sum(avg_probs * K.log(avg_probs + epsilon)))
        
        return quantized

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)

# Calculate vq-vae loss including two VQ stages
def vq_vae_loss_wrapper(data_var, commitment_cost, quantized1, x_inputs1, quantized2, x_inputs2):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat)/data_var
        
        e_latent_loss = K.mean((K.stop_gradient(quantized1) - x_inputs1) ** 2)
        q_latent_loss1 = K.mean((quantized1 - K.stop_gradient(x_inputs1)) ** 2)
        q_latent_loss2 = K.mean((quantized2 - K.stop_gradient(x_inputs2)) ** 2)
        loss = q_latent_loss1 + q_latent_loss2 + commitment_cost * e_latent_loss
        
        return recon_loss + loss
    return vq_vae_loss

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None)

def cb():
    plt.figure(5)
    plt.clf()
    vq1_weights = vqvae.get_layer('vq1').get_weights()[0]
    plt.scatter(vq1_weights[0,:],vq1_weights[1,:], marker='X', color="red")
    if args.vq_stages == 2:
        vq2_weights = vqvae.get_layer('vq2').get_weights()[0]
        plt.scatter(1+vq2_weights[0,:],1+vq2_weights[1,:], marker='x')
    plt.xlim([-1.5,1.5]); plt.ylim([-1.5,1.5])
    plt.draw()
    plt.pause(0.0001)
print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: cb() )

# Hyper Parameters.
batch_size = 64
validation_split = 0.1
commitment_cost = 0.25
train_scale = 0.125
nb_timesteps = 8

parser = argparse.ArgumentParser(description='Two stage VQ-VAE for rate K vectors')
parser.add_argument('featurefile', help='f32 file of eband vectors')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--eband_K', type=int, default=14, help='Length of eband vector')
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of frames to train on')
parser.add_argument('--embedding_dim', type=int, default=2,  help='dimension of embedding vectors')
parser.add_argument('--num_embedding', type=int, default=128,  help='number of embedded vectors')
parser.add_argument('--vq_stages', type=int, default=2,  help='number of vq_stages')
args = parser.parse_args()
dim = args.embedding_dim
nb_samples = args.nb_samples
eband_K = args.eband_K
nb_features = eband_K

# read in rate K vectors ---------------------------------------------------------

features = np.fromfile(args.featurefile, dtype='float32', count = args.nb_samples*eband_K)
nb_samples = int(len(features)/eband_K)
nb_chunks = int(nb_samples/nb_timesteps)
nb_samples = nb_chunks*nb_timesteps
print("nb_samples: %d" % (nb_samples))
features = features[:nb_samples*eband_K].reshape((nb_samples, eband_K))

# normalise
train_mean = np.mean(features, axis=0)
features -= train_mean
features *= train_scale

# reshape into (batch, timesteps, channels) for conv1D
train = features[:nb_samples,:].reshape(nb_chunks, nb_timesteps, eband_K)
for i in range(1,nb_timesteps):
    print(i,-nb_timesteps+i)
    features1 = features[i:nb_samples-nb_timesteps+i,:].reshape(nb_chunks-1, nb_timesteps, eband_K)
    train =  np.concatenate((train,features1))
  
# Model -------------------------------------

# Encoder
input_shape = (nb_timesteps, nb_features)
inputs = Input(shape=input_shape, name='encoder_input')
x = Conv1D(32, 3, activation='tanh', padding='same')(inputs)
x = MaxPooling1D(pool_size=2, padding='same')(x)
x = Conv1D(16, 3, activation='tanh', padding='same')(x)
#x = MaxPooling1D(pool_size=2, padding='same')(x)

encoder = Model(inputs, x)
encoder.summary()

# Two stage Vector Quantiser
vqinit1 = keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
x1 = VQVAELayer(dim, args.num_embedding, commitment_cost, name="vq1", initializer=vqinit1)(x)
x2 = Lambda(lambda x1: x + K.stop_gradient(x1 - x))(x1)

stage1_error = Subtract()([x,x2])
x3 = VQVAELayer(dim, args.num_embedding, commitment_cost, name="vq2")(stage1_error)
x4 = Lambda(lambda x3: stage1_error + K.stop_gradient(x3 - stage1_error))(x3)

x5 = Add()([x2,x4])

#y = UpSampling1D(size=2)(x5)
if args.vq_stages == 1:
    y = Conv1D(16, 3, activation='tanh', padding='same')(x2)
else:
    y = Conv1D(16, 3, activation='tanh', padding='same')(x5)    
y = UpSampling1D(size=2)(y)
y = Conv1D(32, 3, activation='tanh', padding='same')(y)
y = Conv1D(eband_K, 3, padding='same')(y)

vqvae = Model(inputs, y)
data_var = np.var(train)
loss = vq_vae_loss_wrapper(data_var, commitment_cost, x1, x, x3, stage1_error)
adam = keras.optimizers.Adam(lr=0.001)
vqvae.compile(loss=loss, optimizer=adam)
vqvae.summary()
plot_model(vqvae, to_file='vq_vae_conv1d_2stage.png', show_shapes=True)

history = vqvae.fit(train, train,
                    batch_size=batch_size, epochs=args.epochs,
                    validation_split=validation_split,
                    callbacks=[print_weights])

# back to original shape
train_est = vqvae.predict(train, batch_size=batch_size)
encoder_out = encoder.predict(train, batch_size=batch_size)
train = train.reshape(-1, eband_K)
train_est = train_est.reshape(-1, eband_K)
print(encoder_out.shape)
encoder_out = encoder_out.reshape(-1, dim)
print(train.shape, encoder_out.shape)

# Count how many times each vector is used
def vector_count(x, vq, dim, nb_vecs):
    # VQ search outside of Keras Backend
    flat_inputs = np.reshape(x, (-1, dim))
    distances = np.sum(flat_inputs**2, axis=1, keepdims=True) - 2* np.dot(flat_inputs, vq) + np.sum(vq ** 2, axis=0, keepdims=True)
    encoding_indices = np.argmax(-distances, axis=1)
    count = np.zeros(nb_vecs, dtype="int")
    count[encoding_indices] += 1
    return count

vq1_weights = vqvae.get_layer('vq1').get_weights()[0]
count = np.zeros(args.num_embedding, dtype="int")
for i in range(0, nb_samples, batch_size):
    count += vector_count(encoder_out[i:i+batch_size], vq1_weights, dim, args.num_embedding)    

# Plot training results
loss = history.history['loss'] 
val_loss = history.history['val_loss']
num_epochs = range(1, 1 + len(history.history['loss'])) 

plt.figure(1)
plt.plot(num_epochs, loss, label='Training loss')
plt.plot(num_epochs, val_loss, label='Validation loss') 

plt.title('Training and validation loss')
plt.show(block=False)

# Calculate total mean square error and mse per frame

def calc_mse(train, train_est, nb_samples, nb_features, dec):
    msepf = np.zeros(nb_samples-dec)
    e1 = 0; n = 0
    for i in range(nb_samples-dec):
        e = (10*train_est[i,:] - 10*train[i,:])**2
        msepf[i] = np.mean(e)
        e1 += np.sum(e); n += nb_features
    mse = e1/n
    return mse, msepf

mse,msepf = calc_mse(train/train_scale, train_est/train_scale, nb_samples, nb_features, 1)
print("mse: %4.2f dB*dB" % (mse))
plt.figure(2)
plt.plot(msepf)
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
plt.figure(3)
plt.hist(reject_outliers(msepf), bins='fd')

# plot input/output spectra for a few frames to sanity check

nb_plots = 8
frames=range(100,100+nb_plots)
print(frames)
nb_plotsy = np.floor(np.sqrt(nb_plots)); nb_plotsx=nb_plots/nb_plotsy;
plt.figure(4)
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

plt.figure(6)
plt.plot(count,'bo')
plt.title('Vector Usage Counts for Stage 1')
print(count)
plt.show(block=False)

# plot first 2D of spaces

fig,ax = plt.subplots()
ax.hist2d(encoder_out[:,0],encoder_out[:,1], bins=(50,50))
vq1_weights = vqvae.get_layer('vq1').get_weights()[0]
ax.scatter(vq1_weights[0,:],vq1_weights[1,:], marker='X', color="red")
if args.vq_stages == 2:
    vq2_weights = vqvae.get_layer('vq2').get_weights()[0]
    ax.scatter(1+vq2_weights[0,:],1+vq2_weights[1,:], marker='x')
plt.show(block=False)

plt.waitforbuttonpress(0)
plt.close()

