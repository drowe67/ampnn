import tensorflow as tf

'''
TODO
  [ ] get VQ around the right way
  [ ] rename
  [ ] stand alone decoder
  [ ] integrate back into demo, will it operate outside of eager mode?
  [ ] try with two stage/speech data
'''

# VQ layer.
class VQVAELayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_embeddings, commitment_cost,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.initializer = initializer
        self.gamma = 0.99
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = tf.Variable(initial_value=tf.constant([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]]),
                               trainable=False)

        # running sums/EWMA filter states
        self.Centroid_sum = self.w
        self.Centroid_n = tf.Variable(initial_value=tf.ones([self.num_embeddings]), trainable=False)
        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension
        flat_inputs = tf.reshape(x, (-1, self.embedding_dim))
        
        # Calculate distances of input to embedding vectors
        distances = (tf.math.reduce_sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * tf.tensordot(flat_inputs, tf.transpose(self.w), 1)
                     + tf.math.reduce_sum(tf.transpose(self.w) ** 2, axis=0, keepdims=True))
        # Retrieve encoding indices
        encoding_indices = tf.argmax(-distances, axis=1)
        encoding_onehot = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encoding_onehot,self.w)
        
        # Update VQ using EWMA
        centroid_sum =  tf.matmul(tf.transpose(encoding_onehot),x)
        centroid_n = tf.reduce_sum(encoding_onehot,axis=0)
        self.Centroid_sum = self.Centroid_sum*self.gamma + centroid_sum*(1-self.gamma)
        self.Centroid_n = self.Centroid_n*self.gamma + centroid_n*(1-self.gamma)
        print(self.Centroid_sum, self.Centroid_n, tf.reshape(self.Centroid_n, (-1, 1)))
        self.w = self.Centroid_sum/tf.reshape(self.Centroid_n, (-1, 1))
        
        return quantized

    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        encoding_onehot = tf.one_hot(encoding_indices, self.num_embeddings)
        return tf.matmul(encoding_onehot,self.w)

print(tf.test.is_gpu_available())
vq = VQVAELayer(embedding_dim=2,num_embeddings=4,commitment_cost=0.25)
#vq.set_weights(vq_table)

a=tf.constant([[1.,1.],[-1.,1],[1.,1]])
vq(a)

