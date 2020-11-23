'''
TODO:
  [ ] basic QPSK VQ working, like Octave code
  [ ] then hook up to VQVAE, commitment cost etc
'''

import logging
import os
import numpy as np

# Give TF "a bit of shoosh"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

# Custom Layer - Vector Quantiser with Exponential Weighted Moving Average kmeans updates
class VQ_EWMA_kmeans(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, **kwargs):
        self.embedding_dim = embedding_dim          # dimension of each vector
        self.num_embeddings = num_embeddings        # number of VQ entries
        self.vq = tf.Variable(tf.constant([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]]),trainable=False)
        self.gamma = 0.99

        # moving averages used for kmeans update of VQ on each batch
        self.ewma_centroid_sum = tf.Variable(self.vq,trainable=False)
        self.ewma_centroid_n = tf.Variable(initial_value=tf.ones([self.num_embeddings]), trainable=False)

        super(VQ_EWMA_kmeans, self).__init__(**kwargs)

    def build(self, input_shape):
        super(VQ_EWMA_kmeans, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension
        flat_inputs = tf.reshape(x, (-1, self.embedding_dim))
        
        # Calculate distances of input to each VQ entry
        distances = (tf.math.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * tf.tensordot(flat_inputs, tf.transpose(self.vq), 1)
                     + tf.math.reduce_sum(tf.transpose(self.vq) ** 2, axis=0, keepdims=True))
        # Retrieve VQ indices
        encoding_indices = tf.argmax(-distances, axis=1)
        encoding_onehot = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encoding_onehot,self.vq)
        
        # Update moving averages and hence update VQ
        
        centroid_sum =  tf.matmul(tf.transpose(encoding_onehot),x)
        centroid_n = tf.reduce_sum(encoding_onehot,axis=0)
        ewma_centroid_sum = self.ewma_centroid_sum*self.gamma + centroid_sum*(1.-self.gamma)
        ewma_centroid_n = self.ewma_centroid_n*self.gamma + centroid_n*(1.-self.gamma)
        vq = ewma_centroid_sum/tf.reshape(ewma_centroid_n, (-1, 1))

        # this magic needed to store the updated states and avoid the dreaded eager execution explosion
        tf.keras.backend.update(self.ewma_centroid_sum, ewma_centroid_sum)
        tf.keras.backend.update(self.ewma_centroid_n, ewma_centroid_n)
        tf.keras.backend.update(self.vq, vq)
        
        return quantized

    def set_vq(self, vq):
        tf.keras.backend.update(self.vq, vq)
        tf.keras.backend.update(self.ewma_centroid_sum, vq)
        
    def get_vq(self):
        return self.vq

# Simple test model

inputs = tf.keras.layers.Input(shape=(2,))
outputs = VQ_EWMA_kmeans(2,4,name="vq")(inputs)

model = tf.keras.Model(inputs, outputs)
model.compile(loss='mse',optimizer='adam')
model.summary()

vq_initial = np.array([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]])/10
model.get_layer('vq').set_vq(vq_initial)
print(model.get_layer('vq').get_vq().numpy())

x_train=np.ones((100,2),dtype=float);

model.fit(x_train, x_train, batch_size=2, epochs=2)
print(model.get_layer('vq').get_vq().numpy())

