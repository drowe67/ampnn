'''

  Custom Vector Quantiser layer written in tf.keras.  It uses kmeans
  to train, with updates performed on each batch using moving
  averages.

  Refs:
  [1] VQ-VAE_Keras_MNIST_Example.ipynb
      https://colab.research.google.com/github/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb
  [2] "Neural Discrete Representation Learning", Aaron van den Oord etc al, 2018

'''

import tensorflow as tf

# Custom Layer - Vector Quantiser with Exponential Weighted Moving Average kmeans updates
class VQ_EWMA_kmeans(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, **kwargs):
        self.embedding_dim = embedding_dim          # dimension of each vector
        self.num_embeddings = num_embeddings        # number of VQ entries
        self.vq = tf.Variable(tf.zeros(shape=(self.num_embeddings, self.embedding_dim)),trainable=False)
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
