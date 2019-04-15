from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

from gym import spaces

from softlearning.utils.keras import PicklableKerasModel
from .base_preprocessor import BasePreprocessor
from .convnet_preprocessor import (
    convnet as make_convnet, invert_convnet, invert_convnet_v2)


def _softplus_inverse(x):
    """Helper which computes the function inverse of `tf.nn.softplus`."""
    return tf.math.log(tf.math.expm1(x))


def sampling(inputs):
    z_mean, z_log_var = inputs
    batch_size = tf.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.random_normal(shape=(batch_size, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def make_encoder(input_shape,
                 latent_size,
                 *args,
                 **kwargs):
    convnet = make_convnet(input_shape,
                           output_size=2*latent_size,
                           *args,
                           **kwargs)
    shift_and_log_scale_diag = convnet(convnet.inputs)
    shift, log_scale_diag = tf.keras.layers.Lambda(
        lambda shift_and_log_scale_diag: tf.split(
            shift_and_log_scale_diag,
            num_or_size_splits=2,
            axis=-1)
    )(shift_and_log_scale_diag)

    latents = tf.keras.layers.Lambda(
        sampling, output_shape=(latent_size,), name='z'
    )([shift, log_scale_diag])

    encoder = tf.keras.Model(
        convnet.inputs, [shift, log_scale_diag, latents], name='encoder')

    return encoder


def make_decoder(encoder):
    convnet = invert_convnet_v2(encoder.get_layer('convnet'))
    assert (convnet.input.shape.as_list()[-1]
            == encoder.get_layer('convnet').output.shape.as_list()[-1] // 2)
    assert (convnet.output.shape.as_list()
            == encoder.get_layer('convnet').input.shape.as_list())
    decoded_images = convnet(convnet.inputs)
    decoder = tf.keras.Model(
        convnet.inputs, decoded_images, name='decoder')

    return decoder


def make_latent_prior(latent_size):
    prior = tfp.distributions.Independent(
        tfp.distributions.Normal(loc=tf.zeros(latent_size), scale=1),
        reinterpreted_batch_ndims=1)
    return prior


def create_beta_vae(image_shape,
                    output_size,
                    *args,
                    beta=1.0,
                    loss_weight=1.0,
                    name='beta_vae',
                    **kwargs):
    encoder = make_encoder(
        image_shape,
        output_size,
        *args,
        **kwargs)
    decoder = make_decoder(encoder)

    outputs = decoder(encoder(encoder.inputs)[2])
    vae = tf.keras.Model(encoder.inputs, outputs, name=name)
    vae.beta = beta
    vae.loss_weight = loss_weight

    return vae


def create_vae_preprocessor(
        input_shapes,
        image_shape,
        output_size,
        vae,
        name="beta_vae_preprocessor"):

    inputs = [
        tf.keras.layers.Input(shape=input_shape)
        for input_shape in input_shapes
    ]

    concatenated_input = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )(inputs)

    image_size = np.prod(image_shape)
    images_flat, input_raw = tf.keras.layers.Lambda(
        lambda x: [x[..., :image_size], x[..., image_size:]]
    )(concatenated_input)

    images = tf.keras.layers.Reshape(image_shape)(images_flat)

    encoded_images = vae.get_layer('encoder')(images)[0]

    output = tf.keras.layers.Lambda(
        lambda x: tf.concat(x, axis=-1)
    )([encoded_images, input_raw])

    model = PicklableKerasModel(inputs, output, name=name)

    return model


class VAEPreprocessor(BasePreprocessor):
    def __init__(self,
                 observation_space,
                 output_size,
                 image_shape,
                 name='vae_preprocessor',
                 *args,
                 **kwargs):
        super(VAEPreprocessor, self).__init__(observation_space, output_size)
        self.image_shape = image_shape

        assert isinstance(observation_space, spaces.Box)
        input_shapes = (observation_space.shape, )

        self.vae = create_beta_vae(
            image_shape,
            output_size,
            **kwargs)
        self.preprocessor = create_vae_preprocessor(
            input_shapes,
            image_shape,
            output_size,
            self.vae,
            name=name,
        )

    def transform(self, observation):
        transformed = self.preprocessor(observation)
        return transformed

    @property
    def trainable_variables(self):
        return self.vae.trainable_variables
