'''
  Two stage VQ VAE for rate K quantisation
'''

import tensorflow as tf
from vq_kmeans import *

def vqvae_models(nb_timesteps, nb_features, dim, num_embedding):

    x = tf.keras.layers.Input(shape=(nb_timesteps+2, nb_features), name='encoder_input')

    # Encoder
    z_e = tf.keras.layers.Conv1D(32, 3, activation='tanh', padding='valid', name="conv1d_a")(x)
    z_e = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(z_e)
    z_e = tf.keras.layers.Conv1D(dim, 3, activation='tanh', padding='same')(z_e)

    # VQ
    z_q1 = VQ_kmeans(dim, num_embedding, name="vq1")(z_e)
    z_q1_error = tf.keras.layers.Subtract()([z_e,z_q1])
    z_q2 = VQ_kmeans(dim, num_embedding, name="vq2")(z_q1_error)
    z_q = tf.keras.layers.Add()([z_q1,z_q2])
    z_q_ = CopyGradient()([z_q, z_e])

    # Decoder
    p = tf.keras.layers.Conv1D(dim, 3, activation='tanh', padding='same')(z_q_)    
    p = tf.keras.layers.UpSampling1D(size=2)(p)
    p = tf.keras.layers.Conv1D(32, 3, activation='tanh', padding='same')(p)
    p = tf.keras.layers.Conv1D(nb_features, 3, padding='same')(p)

    vqvae = tf.keras.Model(x, p)
    encoder = tf.keras.Model(x, z_e)
    vqvae.add_loss(commitment_loss(z_e, z_q))

    return vqvae, encoder
