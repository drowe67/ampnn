import logging
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings):
        super(MyDenseLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.vq = tf.Variable(tf.constant([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]]),trainable=False)

    def build(self, input_shape):
        super(MyDenseLayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension
        flat_inputs = tf.reshape(x, (-1, self.embedding_dim))
        
        # Calculate distances of input to embedding vectors
        distances = (tf.math.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * tf.tensordot(flat_inputs, tf.transpose(self.vq), 1)
                     + tf.math.reduce_sum(tf.transpose(self.vq) ** 2, axis=0, keepdims=True))
        # Retrieve encoding indices
        encoding_indices = tf.argmax(-distances, axis=1)
        encoding_onehot = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encoding_onehot,self.vq)
        
        return quantized

inputs = tf.keras.layers.Input(shape=(2,))
outputs = MyDenseLayer(2,4)(inputs)

model = tf.keras.Model(inputs, outputs)
model.compile(loss='mse',optimizer='adam')
model.summary()
x_train=np.ones((1000,2),dtype=float);
model.fit(x_train, x_train, batch_size=1, epochs=2)

