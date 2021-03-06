# -*- coding: utf-8 -*-
"""VQ-VAE_Keras_MNIST_Example.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

# VQ-VAE Keras MNIST Example
"""

# Imports.
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist

# Load data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data.
x_test = x_test / np.max(x_train)
x_train = x_train / np.max(x_train)

# Add input channel dimension.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Target dictionary.
target_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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
        return tf.nn.embedding_lookup(w, encoding_indices)

# Calculate vq-vae loss.
def vq_vae_loss_wrapper(data_variance, commitment_cost, quantized, x_inputs):
    def vq_vae_loss(x, x_hat):
        recon_loss = losses.mse(x, x_hat) / data_variance
        
        e_latent_loss = K.mean((K.stop_gradient(quantized) - x_inputs) ** 2)
        q_latent_loss = K.mean((quantized - K.stop_gradient(x_inputs)) ** 2)
        loss = q_latent_loss + commitment_cost * e_latent_loss
        
        return recon_loss + loss #* beta
    return vq_vae_loss

# Hyper Parameters.
epochs = 10 # MAX
batch_size = 64
validation_split = 0.1

# VQ-VAE Hyper Parameters.
embedding_dim = 32 # Length of embedding vectors.
num_embeddings = 128 # Number of embedding vectors (high value = high bottleneck capacity).
commitment_cost = 0.25 # Controls the weighting of the loss terms.

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None)

# Encoder
input_img = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(input_img)
#x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
#x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
#x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.3)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
#x = BatchNormalization()(x)
#x = Dropout(0.4)(x)

# VQVAELayer.
enc = Conv2D(embedding_dim, kernel_size=(1, 1), strides=(1, 1), name="pre_vqvae")(x)
enc_inputs = enc
enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc)

x = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)
data_variance = np.var(x_train)
print(x_train.shape, data_variance)
loss = vq_vae_loss_wrapper(data_variance, commitment_cost, enc, enc_inputs)

# Decoder.
x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = Conv2DTranspose(1, (3, 3))(x)

# Autoencoder.
vqvae = Model(input_img, x)
vqvae.compile(loss=loss, optimizer='adam')
vqvae.summary()
w = vqvae.get_layer('vqvae').get_weights()
#print(w)
#exit()

history = vqvae.fit(x_train, x_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_split=validation_split,
                    callbacks=[esc])

# Plot training results.
loss = history.history['loss'] # Training loss.
val_loss = history.history['val_loss'] # Validation loss.
num_epochs = range(1, 1 + len(history.history['loss'])) # Number of training epochs.

plt.plot(num_epochs, loss, label='Training loss') # Plot training loss.
plt.plot(num_epochs, val_loss, label='Validation loss') # Plot validation loss.

plt.title('Training and validation loss')
plt.legend(loc='best')
plt.show(block=False)

# Show original reconstruction.
n_rows = 4
n_cols = 4 # Must be divisible by 2.
samples_per_col = int(n_cols / 2)
sample_offset = np.random.randint(0, len(x_test) - n_rows - n_cols - 1)
#sample_offset = 0

img_idx = 0
plt.figure(figsize=(n_cols * 2, n_rows * 2))
for i in range(1, n_rows + 1):
    for j in range(1, n_cols + 1, 2):
        idx = n_cols * (i - 1) + j

        # Display original.
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.title.set_text('({:d}) Label: {:s} ->'.format(
            img_idx,
            str(target_dict[np.argmax(y_test[img_idx + sample_offset])])))
        ax.imshow(x_test[img_idx + sample_offset].reshape(28, 28),
                  cmap='gray_r',
                  clim=(0, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction.
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.title.set_text('({:d}) Reconstruction'.format(img_idx))
        ax.imshow(vqvae.predict(
            x_test[img_idx + sample_offset].reshape(-1, 28, 28, 1)).reshape(28, 28),
            cmap='gray_r',
            clim=(0, 1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        img_idx += 1
plt.show()

